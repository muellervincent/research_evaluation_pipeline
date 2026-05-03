"""
Diagnostic logic models.
"""

import json

from ...clients.provider_protocol import ModelProvider
from ...config.execution_settings import DiagnosticAnalysisSettings, DiagnosticDecompositionSettings
from ...config.prompt_registry import PromptRegistry, PromptTemplate
from ...core.domain import PaperContext
from ..protocol import Model
from .schemas import DiagnosticGroup, DiagnosticReport, DiagnosticTaskList


class Decomposition(Model[DiagnosticTaskList]):
    """
    Logic for grouping targeted predictions into batches for diagnostic analysis.
    """

    def __init__(self, settings: DiagnosticDecompositionSettings, prompt_registry: PromptRegistry):
        """
        Initialize the diagnostic decomposition model.

        Args:
            settings: Hyperparameters and strategy for diagnostic decomposition.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(self, prompt_assessment_text: str, details_json: str) -> PromptTemplate:
        """
        Construct the diagnostic decomposition prompt.

        Args:
            prompt_assessment_text: The original raw assessment criteria.
            details_json: The targeted predictions to analyze.

        Returns:
            A formatted PromptTemplate instance.
        """
        return self.prompt_registry.get_prompt(
            f"diagnostic.decomposition.{self.settings.strategy.value}"
        ).format(prompt_assessment_text=prompt_assessment_text, details_json=details_json)

    async def generate(self, provider: ModelProvider, prompt: PromptTemplate) -> DiagnosticTaskList:
        """
        Execute the diagnostic decomposition task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt: The formatted prompt to send.

        Returns:
            A DiagnosticTaskList defining the diagnostic plan.
        """
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=DiagnosticTaskList,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
        )


class Analysis(Model[DiagnosticReport]):
    """
    Logic for analyzing causes of model errors by comparing logic against the source document.
    """

    def __init__(self, settings: DiagnosticAnalysisSettings, prompt_registry: PromptRegistry):
        """
        Initialize the analysis model.

        Args:
            settings: Hyperparameters and strategy for analysis.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(
        self, group: DiagnosticGroup, prompt_assessment_text: str, paper_context: PaperContext
    ) -> tuple[PromptTemplate, PaperContext]:
        """
        Construct the analysis prompt.

        Args:
            group: The logical group of predictions to investigate.
            prompt_assessment_text: The original raw assessment criteria.
            paper_context: The source document context.

        Returns:
            A tuple of the formatted prompt and the paper context.
        """
        tasks = group.model_dump_json(include={"tasks"}, indent=2)
        template = self.prompt_registry.get_prompt(
            f"diagnostic.analysis.{self.settings.strategy.value}"
        ).format(prompt_assessment_text=prompt_assessment_text, tasks=tasks)

        return template, paper_context

    async def generate(
        self, provider: ModelProvider, prompt_data: tuple[PromptTemplate, PaperContext]
    ) -> DiagnosticReport:
        """
        Execute the analysis task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt_data: The formatted prompt and paper context.

        Returns:
            A DiagnosticReport containing diagnostic classifications.
        """
        prompt, paper_context = prompt_data
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=DiagnosticReport,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
            paper_context=paper_context,
        )


class FastDiagnostic(Model[DiagnosticReport]):
    """
    Logic for a single-pass analysis of targeted predictions.
    """

    def __init__(self, settings: DiagnosticAnalysisSettings, prompt_registry: PromptRegistry):
        """
        Initialize the fast diagnostic model.

        Args:
            settings: Hyperparameters and strategy for diagnostic analysis.
            prompt_registry: Access to prompt templates.
        """
        self.settings = settings
        self.prompt_registry = prompt_registry

    def build_prompt(
        self,
        assessment_details: list[dict],
        prompt_assessment_text: str,
        paper_context: PaperContext,
    ) -> tuple[PromptTemplate, PaperContext]:
        """
        Construct the fast diagnostic prompt.

        Args:
            assessment_details: The targeted assessment predictions.
            prompt_assessment_text: The original raw assessment criteria.
            paper_context: The source document context.

        Returns:
            A tuple of the formatted prompt and the paper context.
        """
        assessment_details_json = json.dumps(assessment_details, indent=2)
        template = self.prompt_registry.get_prompt(
            f"diagnostic.fast.{self.settings.strategy.value}"
        ).format(
            prompt_assessment_text=prompt_assessment_text, details_json=assessment_details_json
        )

        return template, paper_context

    async def generate(
        self, provider: ModelProvider, prompt_data: tuple[PromptTemplate, PaperContext]
    ) -> DiagnosticReport:
        """
        Execute the fast diagnostic task via the AI provider.

        Args:
            provider: The LLM API client.
            prompt_data: The formatted prompt and paper context.

        Returns:
            A DiagnosticReport containing diagnostic classifications.
        """
        prompt, paper_context = prompt_data
        return await provider.generate_structured_output(
            model_name=self.settings.model.value,
            prompt_text=prompt.user_text,
            response_model=DiagnosticReport,
            temperature=self.settings.temperature,
            system_instruction=prompt.system_text,
            paper_context=paper_context,
        )
