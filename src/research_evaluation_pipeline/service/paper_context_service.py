"""
Paper Context Service.

Owns the full lifecycle of a PaperContext object: initial creation, PDF bytes loading,
provider file upload, extracted markdown restoration, and API side context caching.

All methods that mutate state are idempotent — calling them multiple times is safe and
incurs only a guard-check overhead on subsequent calls.
"""

from loguru import logger

from ..clients.provider_protocol import ModelProvider
from ..config.execution_settings import PipelineProfile
from ..core.artifact_store import ArtifactStore
from ..core.enums import IngestionMode
from ..core.paper_context import PaperContext
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

    async def build_initial_context(self, paper_stem: str, raw_bytes: bytes) -> PaperContext:
        """
        Creates a minimal PaperContext.

        If the pipeline is in EXTRACTION ingestion mode, attempts to restore the
        previously extracted markdown text from the artifact store so that model
        calls can use text rather than a binary upload.

        No file uploads or API caching occur at this stage.

        Args:
            paper_stem: The identifier for the paper.
            raw_bytes: Optional raw PDF bytes.
        """
        paper_context = PaperContext(paper_stem=paper_stem, raw_bytes=raw_bytes)

        await self.restore_extracted_markdown(paper_context)
        await self.restore_uploaded_file_ids(paper_context)

        return paper_context

    async def restore_extracted_markdown(self, paper_context: PaperContext) -> None:
        """
        Populates the raw_text field of the PaperContext from the artifact store.

        This ensures that if a paper has already been extracted, the subsequent
        pipeline stages use the structured Markdown rather than the original PDF.
        """
        if self._profile.ingestion_mode == IngestionMode.MD:
            extraction_key = self._key_builder.preprocess_extract_key()
            cached_extraction = self._artifact_store.get_artifact(extraction_key)
            if cached_extraction:
                logger.info(
                    f"Restoring extracted markdown for {paper_context.paper_stem} from artifact store."
                )
                paper_context.raw_text = cached_extraction.get("content")

    async def prepare_for_model_execution(
        self, paper_context: PaperContext, model_name: str
    ) -> None:
        """
        Ensures the paper context is ready for direct model interaction.

        If markdown text is available, no file upload is needed.
        Otherwise, ensures the PDF bytes are uploaded.

        Idempotent: repeated calls are no-ops after the first successful preparation.
        """
        if paper_context.raw_text:
            logger.debug(
                f"PaperContext for {paper_context.paper_stem} has markdown text. Skipping PDF upload."
            )
            return

        await self.ensure_file_upload(paper_context, model_name)

    async def ensure_api_cache(self, paper_context: PaperContext, model_name: str) -> None:
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

    async def ensure_file_upload(self, paper_context: PaperContext, model_name: str) -> None:
        """
        Uploads the paper's PDF bytes to the provider's File API if not already uploaded.

        This enables referencing files by ID/URI rather than injecting them as
        raw bytes in every request, optimizing performance and enabling larger files.

        Idempotent: if a file ID is already recorded for the current provider, this is a no-op.
        """
        if hasattr(self._provider, "get_provider_type"):
            provider_name = self._provider.get_provider_type(model_name)
        else:
            provider_name = self._provider.provider_type

        file_id = paper_context.uploaded_file_ids.get(provider_name)

        if file_id:
            logger.debug(f"Validating existing {provider_name} file ID: {file_id}")
            is_valid = False
            if hasattr(self._provider, "validate_file_on_provider"):
                is_valid = await self._provider.validate_file_on_provider(model_name, file_id)
            else:
                is_valid = await self._provider.validate_file(file_id)

            if is_valid:
                return
            else:
                logger.warning(
                    f"File {file_id} no longer valid on {provider_name}. Clearing cache and re-uploading."
                )
                del paper_context.uploaded_file_ids[provider_name]

        filename = f"{paper_context.paper_stem}.pdf"
        logger.info(f"Uploading {filename} to {provider_name} File API...")
        try:
            if hasattr(self._provider, "upload_file_to_provider"):
                file_id = await self._provider.upload_file_to_provider(
                    model_name, paper_context.raw_bytes, filename
                )
            else:
                file_id = await self._provider.upload_file(paper_context.raw_bytes, filename)

            paper_context.uploaded_file_ids[provider_name] = file_id
            logger.success(f"File uploaded to {provider_name}. ID/URI: {file_id}")

            # Persist the new file ID to the artifact store
            upload_key = self._key_builder.paper_upload_key(paper_context.raw_bytes)
            self._artifact_store.save_artifact(upload_key, paper_context.uploaded_file_ids)
        except Exception as error:
            logger.error(f"Failed to upload file to {provider_name}: {error}")

    async def restore_uploaded_file_ids(self, paper_context: PaperContext) -> None:
        """
        Populates the uploaded_file_ids field of the PaperContext from the artifact store.

        This ensures that if a paper has already been uploaded to a provider (e.g. OpenAI),
        subsequent pipeline runs reuse the same file identifier instead of re-uploading.
        """
        upload_key = self._key_builder.paper_upload_key(paper_context.raw_bytes)
        cached_uploads = self._artifact_store.get_artifact(upload_key)
        if cached_uploads:
            logger.info(
                f"Restoring {len(cached_uploads)} uploaded file IDs for {paper_context.paper_stem} from artifact store."
            )
            paper_context.uploaded_file_ids.update(cached_uploads)
