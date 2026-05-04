"""
Preprocess pure logic models.
"""

from typing import Any
from ...clients.provider_protocol import ModelProvider
from ...service.prompt_service import PromptService
from ...config.execution_settings import StepSettings, RefinementSettings
from ..protocol import Model
from .schemas import ExtractionResult, RefinementResult
from ...core.paper_context import PaperContext


class Extraction(Model[ExtractionResult]):
    """
    Logic for converting raw document bytes into structured Markdown text.
    """

    def __init__(self, settings: StepSettings, prompt_service: PromptService):
        """
        Initialize the extraction model.

        Args:
            settings: Hyperparameters and strategy for extraction.
            prompt_service: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_service = prompt_service

    def build_prompt(self) -> Any:
        """
        Construct the extraction prompt.

        Returns:
            The raw PromptTemplate for extraction.
        """
        return self.prompt_service.get_prompt("preprocess.extraction")

    async def generate(
        self, provider: ModelProvider, prompt: Any, paper_context: PaperContext
    ) -> ExtractionResult:
        """
        Execute the text extraction task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt: The formatted prompt to send.
            paper_context: The source document context.

        Returns:
            An ExtractionResult containing the full Markdown text.
        """
        content = await provider.generate_text_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
            paper_context=paper_context,
        )
        return ExtractionResult(content=content, paper_stem=paper_context.paper_stem)


class Refinement(Model[RefinementResult]):
    """
    Logic for cleaning and structuring assessment criteria.
    """

    def __init__(self, settings: RefinementSettings, prompt_service: PromptService):
        """
        Initialize the refinement model.

        Args:
            settings: Hyperparameters and strategy for refinement.
            prompt_service: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_service = prompt_service

    def build_prompt(self, prompt_text: str) -> Any:
        """
        Construct the refinement prompt.

        Args:
            prompt_text: The raw assessment criteria.

        Returns:
            A formatted PromptTemplate instance.
        """
        return self.prompt_service.get_prompt(
            f"preprocess.refine.{self.settings.strategy.value}"
        ).format(prompt_master_text=prompt_text)

    async def generate(self, provider: ModelProvider, prompt: Any) -> RefinementResult:
        """
        Execute the prompt refinement task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt: The formatted prompt to send.

        Returns:
            A RefinementResult containing the structured criteria.
        """
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=RefinementResult,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
        )
