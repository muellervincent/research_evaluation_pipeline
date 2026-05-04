"""
Diagnostic orchestration logic.
"""

import json


from loguru import logger

from ...clients.provider_protocol import ModelProvider
from ...config.execution_settings import DiagnosticProfile
from ...config.prompt_registry import PromptRegistry
from ...core.paper_context import PaperContext
from .models import Decomposition, Analysis, FastDiagnostic
from .schemas import DiagnosticReport, DiagnosticGroup, DiagnosticTaskList


class DiagnosticLogic:
    """
    Coordinator for atomic diagnostic transformations.

    Provides a clean interface for executing each step of the diagnostic pipeline
    by managing the diagnostic logic models.
    """

    def __init__(self, provider: ModelProvider, profile: DiagnosticProfile, prompt_registry: PromptRegistry):
        """
        Initialize the diagnostic logic with necessary providers and settings.

        Args:
            provider: The LLM API client.
            profile: The diagnostic-specific execution profile.
            prompt_registry: The repository of prompt templates.
        """
        self.provider = provider
        self.profile = profile
        self.prompt_registry = prompt_registry

        self.decomposition = Decomposition(profile.decomposition, prompt_registry)
        self.analysis = Analysis(profile.analysis, prompt_registry)
        self.fast_logic = FastDiagnostic(profile.analysis, prompt_registry)

    async def decompose(self, assessment_details: list[dict], prompt_assessment_text: str) -> DiagnosticTaskList:
        """
        Execute the decomposition step to create a structured diagnostic task list.

        Args:
            assessment_details: The targeted assessment results to analyze.
            prompt_assessment_text: The original raw assessment criteria.

        Returns:
            The generated DiagnosticTaskList.
        """
        logger.info("Executing diagnostic decomposition...")
        assessment_details_json = json.dumps(assessment_details, indent=2)
        prompt = self.decomposition.build_prompt(prompt_assessment_text, assessment_details_json)
        return await self.decomposition.generate(self.provider, prompt)

    async def analyze_group(
        self, group: DiagnosticGroup, prompt_assessment_text: str, paper_context: PaperContext
    ) -> DiagnosticReport:
        """
        Execute the analysis step to find the causes of model errors.

        Args:
            group: The logical group of diagnostic tasks.
            prompt_assessment_text: The original raw assessment criteria.
            paper_context: The source document context.

        Returns:
            The generated DiagnosticReport containing classifications.
        """
        logger.info(f"Executing diagnostic analysis for group '{group.group_name}'...")
        prompt_data = self.analysis.build_prompt(group, prompt_assessment_text, paper_context)
        return await self.analysis.generate(self.provider, prompt_data)

    async def fast_diagnose(
        self, assessment_details: list[dict], prompt_assessment_text: str, paper_context: PaperContext
    ) -> DiagnosticReport:
        """
        Execute the single-pass fast diagnostic mode.

        Args:
            assessment_details: The targeted assessment results to analyze.
            prompt_assessment_text: The original raw assessment criteria.
            paper_context: The source document context.

        Returns:
            The generated DiagnosticReport.
        """
        logger.info("Executing fast diagnostic...")
        if not assessment_details:
            return DiagnosticReport(analyses=[])

        prompt_data = self.fast_logic.build_prompt(assessment_details, prompt_assessment_text, paper_context)
        return await self.fast_logic.generate(self.provider, prompt_data)
