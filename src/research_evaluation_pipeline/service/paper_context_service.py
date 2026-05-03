"""
Paper Context Service.

Owns the full lifecycle of a PaperContext object: initial creation, PDF bytes loading,
provider file upload, extracted markdown restoration, and API side context caching.

All methods that mutate state are idempotent — calling them multiple times is safe and
incurs only a guard-check overhead on subsequent calls.
"""

from pathlib import Path

from loguru import logger

from ..clients.provider_protocol import ModelProvider
from ..config.execution_settings import PipelineProfile
from ..core.artifact_store import ArtifactStore
from ..core.domain import PaperContext
from ..core.enums import IngestionMode
from .artifact_key_builder import ArtifactKeyBuilder


class PaperContextService:
    """
    Manages the paper context lifecycle in a lazy, idempotent manner.

    Responsibilities:
    - Building the initial minimal PaperContext (markdown restoration only).
    - Preparing bytes and provider file upload on demand, only for steps that need it.
    - Creating the provider-side API context cache on demand, only for steps that use it.

    Steps that are purely prompt-level (refinement, decomposition) never call this service
    after initial context creation, avoiding unnecessary uploads and API cache creation.
    """

    def __init__(
        self,
        provider: ModelProvider,
        artifact_store: ArtifactStore,
        key_builder: ArtifactKeyBuilder,
        profile: PipelineProfile,
    ):
        self._provider = provider
        self._artifact_store = artifact_store
        self._key_builder = key_builder
        self._profile = profile

    async def build_initial_context(self, pdf_path: Path) -> PaperContext:
        """
        Creates a minimal PaperContext.

        If the pipeline is in EXTRACTION ingestion mode, attempts to restore the
        previously extracted markdown text from the artifact store so that model
        calls can use text rather than a binary upload.

        No binary loading, file uploads, or Application Programming Interface caching occur at this stage.

        Args:
            pdf_path: The filesystem path to the source PDF.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_context = PaperContext(pdf_path=pdf_path, ingestion_mode=self._profile.ingestion_mode)

        if paper_context.ingestion_mode == IngestionMode.EXTRACTION:
            extraction_key = self._key_builder.preprocess_extract_key()
            cached_extraction = self._artifact_store.get_artifact(extraction_key)
            if cached_extraction:
                logger.info(
                    f"Restored extracted markdown for {paper_context.paper_stem} from artifact store."
                )
                paper_context.raw_text = cached_extraction.get("content")

        return paper_context

    async def prepare_for_model_execution(self, paper_context: PaperContext) -> None:
        """
        Ensures the paper context is ready for direct model interaction.

        Loads PDF bytes from disk if not already present.
        Uploads the file to the provider's File Application Programming Interface if the ingestion mode is UPLOAD
        and no upload reference exists yet.

        Idempotent: repeated calls are no-ops after the first successful preparation.
        """
        if paper_context.raw_bytes is None:
            logger.debug(f"Loading PDF bytes from {paper_context.pdf_path}")
            paper_context.raw_bytes = Path(paper_context.pdf_path).read_bytes()

        if (
            paper_context.ingestion_mode == IngestionMode.UPLOAD
            and paper_context.uploaded_file is None
        ):
            logger.debug(
                f"Uploading {paper_context.pdf_path} to provider File Application Programming Interface"
            )
            paper_context.uploaded_file = await self._provider.upload_file(
                str(paper_context.pdf_path)
            )

    async def ensure_application_programming_interface_cache(
        self, paper_context: PaperContext, model_name: str
    ) -> None:
        """
        Creates a provider-side context cache entry for the paper if one does not exist.

        The context cache accelerates repeated model calls against the same paper content
        (e.g., all evidence extraction groups for a single paper).

        Idempotent: if a cache name is already recorded on the context, this is a no-op.
        """
        if paper_context.has_model_cache(model_name):
            logger.debug(
                f"API context cache already present for {paper_context.paper_stem} on model {model_name}, skipping."
            )
            return

        logger.info(
            f"Creating API context cache for {paper_context.paper_stem} on model {model_name}"
        )
        await self._provider.cache_content(model_name, paper_context)
