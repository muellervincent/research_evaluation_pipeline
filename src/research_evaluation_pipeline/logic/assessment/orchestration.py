"""
Assessment orchestration logic.
"""


from loguru import logger

from ...clients.provider_protocol import ModelProvider
from ...service.prompt_service import PromptService
from ...config.execution_settings import AssessmentProfile
from ...core.paper_context import PaperContext
from .models import (
    Decomposition,
    Extraction,
    Synthesis,
    FastAssessment
)
from .schemas import AssessmentGroup, AssessmentTaskList, AssessmentEvidenceReport, AssessmentReport


class AssessmentLogic:
    """
    Coordinator for atomic assessment transformations.

    Provides a clean interface for executing each step of the assessment pipeline
    by managing the lifecycle of specific logic models.
    """

    def __init__(self, provider: ModelProvider, profile: AssessmentProfile, prompt_service: PromptService):
        """
        Initialize the assessment logic with necessary providers and settings.

        Args:
            provider: The LLM API client.
            profile: The assessment-specific execution profile.
            prompt_service: The repository of prompt templates.
        """
        self.provider = provider
        self.profile = profile
        self.decomposition = Decomposition(profile.decomposition, prompt_service)
        self.extraction = Extraction(profile.extraction, prompt_service)
        self.synthesis = Synthesis(profile.synthesis, prompt_service)
        self.fast_logic = FastAssessment(profile.synthesis, prompt_service)

    async def decompose(self, refined_prompt: str) -> AssessmentTaskList:
        """
        Execute the decomposition step to create a structured task list.

        Args:
            refined_prompt: The cleaned assessment criteria.

        Returns:
            The generated AssessmentTaskList.
        """
        logger.info("Executing assessment decomposition...")
        prompt = self.decomposition.build_prompt(refined_prompt)
        return await self.decomposition.generate(self.provider, prompt)

    async def extract_evidence(self, group: AssessmentGroup, paper_context: PaperContext) -> AssessmentEvidenceReport:
        """
        Execute the evidence extraction step for a specific group.

        Args:
            group: The logical group of assessment tasks.
            paper_context: The source document context.

        Returns:
            The generated AssessmentEvidenceReport.
        """
        logger.info("Executing assessment extraction for group...")
        prompt_data = self.extraction.build_prompt(group, paper_context)
        return await self.extraction.generate(self.provider, prompt_data)

    async def synthesize_report(self, group: AssessmentGroup, evidence: AssessmentEvidenceReport) -> AssessmentReport:
        """
        Execute the synthesis step to render decisions from evidence.

        Args:
            group: The logical group of assessment tasks.
            evidence: The evidence report for this group.

        Returns:
            The generated AssessmentReport.
        """
        logger.info(f"Executing assessment synthesis for group {group.group_name} with strategy {self.profile.synthesis.strategy.value}...")
        prompt_data = self.synthesis.build_prompt(group, evidence)
        return await self.synthesis.generate(self.provider, prompt_data)

    async def execute_fast(self, refined_prompt: str, paper_context: PaperContext) -> AssessmentReport:
        """
        Execute the single-pass fast assessment mode.

        Args:
            refined_prompt: The cleaned assessment criteria.
            paper_context: The source document context.

        Returns:
            The generated AssessmentReport.
        """
        logger.info(f"Executing FAST assessment synthesis with strategy {self.profile.synthesis.strategy.value}...")
        prompt_data = self.fast_logic.build_prompt(refined_prompt, paper_context)
        return await self.fast_logic.generate(self.provider, prompt_data)
