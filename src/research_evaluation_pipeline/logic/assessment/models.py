"""
Assessment logic models.
Atomic building blocks for assessment transformations.
"""

from ...clients.provider_protocol import ModelProvider
from ...config.prompt_registry import PromptRegistry, PromptTemplate
from ...config.execution_settings import (
    AssessmentDecompositionSettings,
    AssessmentExtractionSettings,
    AssessmentSynthesisSettings
)
from ..protocol import Model
from .schemas import (
    AssessmentReport,
    AssessmentEvidenceReport,
    AssessmentGroup,
    AssessmentTaskList,
)
from ...core.domain import PaperContext


class Decomposition(Model[AssessmentTaskList]):
    """
    Logic for breaking down assessment criteria into cohesive task groups.
    """

    def __init__(self, settings: AssessmentDecompositionSettings, prompt_registry: PromptRegistry):
        """
        Initialize the decomposition model with specific strategy and settings.

        Args:
            settings: Hyperparameters and strategy for decomposition.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(self, refined_prompt_text: str) -> PromptTemplate:
        """
        Construct the prompt for criteria decomposition.

        Args:
            refined_prompt_text: The cleaned assessment criteria.

        Returns:
            A formatted PromptTemplate instance.
        """
        return self.prompt_registry.get_prompt(
            f"assessment.decomposition.{self.settings.strategy.value}"
        ).format(prompt_refined_text=refined_prompt_text)

    async def generate(self, provider: ModelProvider, prompt: PromptTemplate) -> AssessmentTaskList:
        """
        Execute the decomposition task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt: The formatted prompt to send.

        Returns:
            An AssessmentTaskList defining the task groups.
        """
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=AssessmentTaskList,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
        )


class Extraction(Model[AssessmentEvidenceReport]):
    """
    Logic for locating and extracting verbatim quotes from a document for assessment.
    """

    def __init__(self, settings: AssessmentExtractionSettings, prompt_registry: PromptRegistry):
        """
        Initialize the extraction model.

        Args:
            settings: Hyperparameters and strategy for extraction.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(self, group: AssessmentGroup, paper_context: PaperContext) -> tuple[PromptTemplate, PaperContext]:
        """
        Construct the extraction prompt for a specific task group.

        Args:
            group: The logical group of questions to answer.
            paper_context: The source document context.

        Returns:
            A tuple of the formatted prompt and the paper context.
        """
        questions = "\n".join([f"- {t.question_id}: {t.question_text}" for t in group.tasks])
        template = self.prompt_registry.get_prompt(
            f"assessment.extraction.{self.settings.strategy.value}"
        ).format(group_name=group.group_name, questions=questions)

        return template, paper_context

    async def generate(self, provider: ModelProvider, prompt_data: tuple[PromptTemplate, PaperContext]) -> AssessmentEvidenceReport:
        """
        Execute the evidence extraction task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt_data: The formatted prompt and paper context.

        Returns:
            An AssessmentEvidenceReport containing found quotes.
        """
        prompt, paper_context = prompt_data
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=AssessmentEvidenceReport,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
            paper_context=paper_context,
        )


class Synthesis(Model[AssessmentReport]):
    """
    Logic for rendering final assessment decisions based on extracted evidence.
    """

    def __init__(self, settings: AssessmentSynthesisSettings, prompt_registry: PromptRegistry):
        """
        Initialize the synthesis model.

        Args:
            settings: Hyperparameters and strategy for synthesis.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(self, group: AssessmentGroup, evidence: AssessmentEvidenceReport) -> PromptTemplate:
        """
        Construct the synthesis prompt using previously extracted evidence.

        Args:
            group: The logical group of questions.
            evidence: The evidence report for this group.

        Returns:
            A formatted PromptTemplate instance.
        """
        criteria = "\n".join([f"{t.question_id}. {t.question_text}" for t in group.tasks])
        return self.prompt_registry.get_prompt(
            f"assessment.synthesis.{self.settings.strategy.value}"
        ).format(prompt_refined_text=criteria, evidence_json=evidence.model_dump_json(indent=4))

    async def generate(self, provider: ModelProvider, prompt: PromptTemplate) -> AssessmentReport:
        """
        Execute the synthesis task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt: The formatted prompt to send.

        Returns:
            An AssessmentReport containing final decisions.
        """
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=AssessmentReport,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
        )


class FastAssessment(Model[AssessmentReport]):
    """
    Logic for a single-pass assessment of the entire document.
    """

    def __init__(self, settings: AssessmentSynthesisSettings, prompt_registry: PromptRegistry):
        """
        Initialize the fast assessment model.

        Args:
            settings: Hyperparameters and strategy for assessment.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(self, refined_prompt: str, paper_context: PaperContext) -> tuple[PromptTemplate, PaperContext]:
        """
        Construct the single-pass assessment prompt.

        Args:
            refined_prompt: The cleaned assessment criteria.
            paper_context: The source document context.

        Returns:
            A tuple of the formatted prompt and the paper context.
        """
        template = self.prompt_registry.get_prompt(f"assessment.fast.{self.settings.strategy.value}").format(
            prompt_refined_text=refined_prompt
        )
        return template, paper_context

    async def generate(self, provider: ModelProvider, prompt_data: tuple[PromptTemplate, PaperContext]) -> AssessmentReport:
        """
        Execute the fast assessment task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt_data: The formatted prompt and paper context.

        Returns:
            An AssessmentReport containing final decisions.
        """
        prompt, paper_context = prompt_data
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=AssessmentReport,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
            paper_context=paper_context,
        )
