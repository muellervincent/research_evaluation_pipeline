"""
Preprocess orchestration logic.
Coordinates concurrent document ingestion and prompt refinement.
"""

from loguru import logger

from research_evaluation_pipeline.logic.preprocess.schemas import ExtractionResult, RefinementResult

from ...clients.provider_protocol import ModelProvider
from ...config.execution_settings import PreprocessProfile
from ...service.prompt_service import PromptService
from ...core.paper_context import PaperContext
from .models import Extraction, Refinement


class PreprocessLogic:
    """
    Coordinator for atomic preprocess transformations.

    Provides a clean interface for executing document extraction and criteria
    refinement by managing the lifecycle of specific logic models.
    """

    def __init__(
        self, provider: ModelProvider, profile: PreprocessProfile, prompt_service: PromptService
    ):
        """
        Initialize the preprocess logic with necessary providers and settings.

        Args:
            provider: The LLM API client.
            profile: The preprocess-specific execution profile.
            prompt_service: The repository of prompt templates.
        """
        self.provider = provider
        self.profile = profile
        self.extraction = Extraction(profile.extraction, prompt_service)
        self.refinement = Refinement(profile.refinement, prompt_service)

    async def refine_prompt(self, master_prompt: str) -> RefinementResult:
        """
        Execute the prompt refinement step to clean and structure the criteria.

        Args:
            master_prompt: The raw assessment criteria from the filesystem.

        Returns:
            The generated RefinementResult.
        """
        logger.info("Executing prompt refinement...")
        prompt = self.refinement.build_prompt(master_prompt)
        return await self.refinement.generate(self.provider, prompt)

    async def extract_paper(self, paper_context: PaperContext) -> ExtractionResult:
        """
        Execute the document extraction step to convert PDF to Markdown.

        Args:
            paper_context: The source document context.

        Returns:
            The generated ExtractionResult.
        """
        logger.info(f"Executing extraction for {paper_context.paper_stem}")
        prompt = self.extraction.build_prompt()
        return await self.extraction.generate(self.provider, prompt, paper_context)
