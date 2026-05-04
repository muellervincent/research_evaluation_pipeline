"""
Master Orchestrator.

Coordinates the research assessment and diagnostic pipeline by delegating to atomic model-facing
logic classes. All infrastructure concerns (paper lifecycle, artifact keys, concurrency,
dependency resolution) are handled by injected services.
"""

from typing import Awaitable, TypeVar

from loguru import logger

from ..clients.provider_protocol import ModelProvider
from ..config.execution_settings import PipelineProfile
from ..config.prompt_registry import PromptRegistry
from ..core.artifact_store import ArtifactStore
from ..core.enums import CachePolicy
from ..core.paper_context import PaperContext
from ..logic.assessment.orchestration import AssessmentLogic
from ..logic.assessment.schemas import (
    AssessmentEvidenceReport,
    AssessmentGroup,
    AssessmentReport,
    AssessmentTaskList,
)
from ..logic.diagnostic.orchestration import DiagnosticLogic
from ..logic.diagnostic.schemas import DiagnosticGroup, DiagnosticReport, DiagnosticTaskList
from ..logic.preprocess.orchestration import PreprocessLogic
from ..logic.preprocess.schemas import ExtractionResult, RefinementResult
from .artifact_key_builder import ArtifactKeyBuilder
from .paper_context_service import PaperContextService
from .step_executor import StepExecutor

T = TypeVar("T")


class MasterOrchestrator:
    """
    State machine that runs the research assessment and diagnostic DAG.

    Responsibilities:
    - Delegating each pipeline step to the appropriate logic class.
    - Wrapping each step with artifact caching via _execute_with_cache.
    - Coordinating multi-step stages (preprocess, assessment, diagnostic).

    All infrastructure concerns are handled by injected services:
    - ArtifactKeyBuilder: deterministic cache key construction.
    - PaperContextService: paper bytes, uploads, and API side context caching.
    - StepExecutor: dependency resolution, group dispatch, and concurrency.
    """

    def __init__(
        self,
        provider: ModelProvider,
        profile: PipelineProfile,
        prompt_registry: PromptRegistry,
        artifact_store: ArtifactStore,
        key_builder: ArtifactKeyBuilder,
        paper_context_service: PaperContextService,
        step_executor: StepExecutor,
    ):
        """
        Initialize the orchestrator with all necessary services and configuration.

        Args:
            provider: The LLM API client.
            profile: The execution settings for this run.
            prompt_registry: The repository of prompt templates.
            artifact_store: The persistent cache for results.
            key_builder: The generator for deterministic artifact keys.
            paper_context_service: The service managing document lifecycle.
            step_executor: The engine for dispatching pipeline steps.
        """
        self.provider = provider
        self.profile = profile
        self.prompt_registry = prompt_registry
        self.artifact_store = artifact_store
        self.key_builder = key_builder
        self.paper_context_service = paper_context_service
        self.step_executor = step_executor

        self.preprocess = PreprocessLogic(provider, profile.preprocess, prompt_registry)
        self.assessment = AssessmentLogic(provider, profile.assessment, prompt_registry)
        if profile.diagnostic:
            self.diagnostic = DiagnosticLogic(provider, profile.diagnostic, prompt_registry)
        else:
            self.diagnostic = None

        self.step_executor.bind_orchestrator(self)

    async def _execute_with_cache(
        self, key: str, policy: CachePolicy, response_model: type[T], coroutine: Awaitable[T]
    ) -> T:
        """
        Execute an awaitable with integrated ArtifactStore caching.

        If the cache policy is USE_CACHE and a hit is found, the coroutine is cancelled
        and the cached value is returned. Otherwise, the task is executed, and its
        result is persisted to both the run history and (optionally) the active cache.

        Args:
            key: The unique string identifier for the artifact.
            policy: The rule for cache interaction (USE, BYPASS, OVERWRITE).
            response_model: The Pydantic model for validation.
            coroutine: The asynchronous task to execute on a cache miss.

        Returns:
            The validated result object of type T.
        """
        if policy == CachePolicy.USE_CACHE:
            cached = self.artifact_store.get_artifact(key)
            if cached:
                logger.debug(f"Cache hit for {key}. Skipping execution.")
                if hasattr(coroutine, "close"):
                    coroutine.close()
                return response_model.model_validate(cached)

        logger.debug(f"Executing step for key: {key}")
        result = await coroutine

        json_content = result.model_dump(mode="json")
        self.artifact_store.save_run(key, json_content)

        if policy in (CachePolicy.USE_CACHE, CachePolicy.OVERWRITE_CACHE):
            self.artifact_store.save_artifact(key, json_content)

        return result

    async def execute_preprocess_refine(self, master_prompt: str) -> RefinementResult:
        """
        Clean and structure the master criteria.

        Args:
            master_prompt: The raw criteria text from the filesystem.

        Returns:
            A RefinementResult containing the cleaned prompt and optional ID mappings.
        """
        return await self._execute_with_cache(
            key=self.key_builder.preprocess_refine_key(),
            policy=self.profile.preprocess.refinement.cache_policy,
            response_model=RefinementResult,
            coroutine=self.preprocess.refine_prompt(master_prompt),
        )

    async def execute_preprocess_extraction(self, paper_context: PaperContext) -> ExtractionResult:
        """
        Convert the source PDF into a structured Markdown document.

        Args:
            paper_context: The context object managing the paper's lifecycle.

        Returns:
            An ExtractionResult containing the full Markdown text.
        """
        return await self._execute_with_cache(
            key=self.key_builder.preprocess_extract_key(),
            policy=self.profile.preprocess.extraction.cache_policy,
            response_model=ExtractionResult,
            coroutine=self.preprocess.extract_paper(paper_context),
        )

    async def execute_assessment_extraction(
        self, group: AssessmentGroup, paper_context: PaperContext
    ) -> AssessmentEvidenceReport:
        """
        Locate and extract verbatim evidence for a specific assessment group.

        Args:
            group: The logical group of questions to investigate.
            paper_context: The source document context.

        Returns:
            An AssessmentEvidenceReport containing found quotes.
        """
        return await self._execute_with_cache(
            key=self.key_builder.assessment_extract_key(group),
            policy=self.profile.assessment.extraction.cache_policy,
            response_model=AssessmentEvidenceReport,
            coroutine=self.assessment.extract_evidence(group, paper_context),
        )

    async def execute_assessment_synthesis(
        self, group: AssessmentGroup, evidence: AssessmentEvidenceReport
    ) -> AssessmentReport:
        """
        Determine final assessment decisions for a specific group based on extracted evidence.

        Args:
            group: The logical group of questions.
            evidence: The evidence report previously extracted for this group.

        Returns:
            An AssessmentReport containing boolean answers and justifications.
        """
        return await self._execute_with_cache(
            key=self.key_builder.assessment_synthesize_key(group.group_name, evidence),
            policy=self.profile.assessment.synthesis.cache_policy,
            response_model=AssessmentReport,
            coroutine=self.assessment.synthesize_report(group, evidence),
        )

    async def execute_fast_assessment(
        self, paper_context: PaperContext, refined_prompt: str
    ) -> AssessmentReport:
        """
        Run a single-pass assessment of the entire document.

        Args:
            paper_context: The source document context.
            refined_prompt: The cleaned assessment criteria.

        Returns:
            An AssessmentReport containing final decisions.
        """
        return await self._execute_with_cache(
            key=self.key_builder.assessment_fast_key(refined_prompt),
            policy=self.profile.assessment.synthesis.cache_policy,
            response_model=AssessmentReport,
            coroutine=self.assessment.execute_fast(refined_prompt, paper_context),
        )

    async def execute_assessment_decomposition(self, refined_prompt: str) -> AssessmentTaskList:
        """
        Decompose assessment criteria into logically cohesive task groups.

        Args:
            refined_prompt: The cleaned assessment criteria.

        Returns:
            An AssessmentTaskList defining the execution plan.
        """
        return await self._execute_with_cache(
            key=self.key_builder.assessment_decompose_key(refined_prompt),
            policy=self.profile.assessment.decomposition.cache_policy,
            response_model=AssessmentTaskList,
            coroutine=self.assessment.decompose(refined_prompt),
        )

    async def execute_diagnostic_decomposition(
        self, filtered_details: list[dict], assessment_prompt: str
    ) -> DiagnosticTaskList:
        """
        Group model predictions into batches for detailed diagnostic analysis.

        Args:
            filtered_details: A list of assessment results targeted for analysis.
            assessment_prompt: The original assessment criteria text.

        Returns:
            A DiagnosticTaskList defining the diagnostic plan.
        """
        return await self._execute_with_cache(
            key=self.key_builder.diagnostic_decompose_key(assessment_prompt, filtered_details),
            policy=self.profile.diagnostic.decomposition.cache_policy,
            response_model=DiagnosticTaskList,
            coroutine=self.diagnostic.decompose(filtered_details, assessment_prompt),
        )

    async def execute_diagnostic_analysis(
        self, group: DiagnosticGroup, assessment_prompt: str, paper_context: PaperContext
    ) -> DiagnosticReport:
        """
        Analyze a diagnostic group to find the cause of errors.

        Args:
            group: The diagnostic task group to analyze.
            assessment_prompt: The original assessment criteria text.
            paper_context: The source document context.

        Returns:
            A DiagnosticReport containing diagnostic classifications.
        """
        return await self._execute_with_cache(
            key=self.key_builder.diagnostic_analyze_key(group, assessment_prompt),
            policy=self.profile.diagnostic.analysis.cache_policy,
            response_model=DiagnosticReport,
            coroutine=self.diagnostic.analyze_group(group, assessment_prompt, paper_context),
        )

    async def execute_fast_diagnostic(
        self, filtered_details: list[dict], assessment_prompt: str, paper_context: PaperContext
    ) -> DiagnosticReport:
        """
        Run a single-pass analysis of targeted model predictions.

        Args:
            filtered_details: The predictions targeted for analysis.
            assessment_prompt: The original assessment criteria text.
            paper_context: The source document context.

        Returns:
            A DiagnosticReport containing diagnostic classifications.
        """
        return await self._execute_with_cache(
            key=self.key_builder.diagnostic_fast_key(assessment_prompt, filtered_details),
            policy=self.profile.diagnostic.analysis.cache_policy,
            response_model=DiagnosticReport,
            coroutine=self.diagnostic.fast_diagnose(
                filtered_details, assessment_prompt, paper_context
            ),
        )

    async def reconstruct_assessment_report(self) -> AssessmentReport | None:
        """
        Assemble a complete AssessmentReport from cached fragment synthesis artifacts.

        Returns:
            The merged AssessmentReport if all fragments are present in cache, else None.
        """
        refinement_result = self.step_executor.require_refinement_result()
        prompt = refinement_result.refined_prompt

        task_list_data = self.artifact_store.get_artifact(
            self.key_builder.assessment_decompose_key(prompt)
        )
        if not task_list_data:
            return None

        task_list = AssessmentTaskList(**task_list_data)
        all_answers = []

        for group in task_list.groups:
            evidence_data = self.artifact_store.get_artifact(
                self.key_builder.assessment_extract_key(group)
            )
            if not evidence_data:
                logger.warning(f"Missing evidence artifact for group '{group.group_name}'.")
                return None
            evidence = AssessmentEvidenceReport(**evidence_data)

            synthesis_data = self.artifact_store.get_artifact(
                self.key_builder.assessment_synthesize_key(group.group_name, evidence)
            )
            if synthesis_data:
                report = AssessmentReport(**synthesis_data)
                all_answers.extend(report.answers)
            else:
                logger.warning(f"Missing synthesis artifact for group '{group.group_name}'.")
                return None

        return AssessmentReport(answers=all_answers)

    async def reconstruct_diagnostic_report(
        self, prompt: str, assessment_details: list[dict]
    ) -> DiagnosticReport | None:
        """
        Assemble a complete DiagnosticReport from cached fragment analysis artifacts.

        Args:
            prompt: The assessment criteria prompt.
            assessment_details: The filtered assessment details targeted for diagnostic.

        Returns:
            The merged DiagnosticReport if all fragments are present in cache, else None.
        """
        task_list_data = self.artifact_store.get_artifact(
            self.key_builder.diagnostic_decompose_key(prompt, assessment_details)
        )
        if not task_list_data:
            return None

        task_list = DiagnosticTaskList(**task_list_data)
        all_analyses = []

        for group in task_list.groups:
            analysis_data = self.artifact_store.get_artifact(
                self.key_builder.diagnostic_analyze_key(group, prompt)
            )
            if analysis_data:
                report = DiagnosticReport(**analysis_data)
                all_analyses.extend(report.analyses)
            else:
                logger.warning(f"Missing analysis artifact for group '{group.group_name}'.")
                return None

        return DiagnosticReport(analyses=all_analyses)
