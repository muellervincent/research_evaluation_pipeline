"""
Step Executor.

Owns all cross-cutting concerns that sit between the CLI and the model-facing orchestrator:
- Artifact dependency resolution (loading prerequisites from the artifact store).
- Group iteration and concurrency dispatch (concurrent vs sequential).
- Paper Context mapping utilities for diagnostic (ID resolution, detail filtering).

Both the MasterOrchestrator (for full-stage runs) and the CLI (for atomic step runs)
delegate to this class, ensuring identical behaviour at every abstraction level.
"""

import asyncio
from typing import Awaitable, TypeVar

from loguru import logger

from ..config.execution_settings import PipelineProfile
from ..core.artifact_store import ArtifactStore
from ..core.enums import DiagnosticAnalysisStrategy, DiagnosticPromptSource, RefinementStrategy
from ..core.paper_context import PaperContext
from ..logic.assessment.schemas import (
    AssessmentAnswer,
    AssessmentEvidenceReport,
    AssessmentGroup,
    AssessmentTaskList,
)
from ..logic.diagnostic.schemas import DiagnosticGroup, DiagnosticItem, DiagnosticTaskList
from ..logic.preprocess.schemas import RefinementResult
from .artifact_key_builder import ArtifactKeyBuilder

TaskResultType = TypeVar("TaskResultType")


class StepExecutor:
    """
    Handles artifact dependency resolution, group iteration, and concurrency dispatch.

    The MasterOrchestrator calls the dispatch methods from within full-stage pipeline
    methods. The CLI calls the require_* methods directly to validate dependencies before
    delegating to the orchestrator for atomic step execution.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        key_builder: ArtifactKeyBuilder,
        profile: PipelineProfile,
    ):
        """
        Initialize the step executor with necessary data access services.

        Args:
            artifact_store: The persistent storage for pipeline results.
            key_builder: The generator for deterministic artifact keys.
            profile: The configuration for the current pipeline run.
        """
        self._artifact_store = artifact_store
        self._key_builder = key_builder
        self._profile = profile

        self._orchestrator = None

    def bind_orchestrator(self, orchestrator) -> None:
        """
        Bind the MasterOrchestrator instance to resolve circular dependencies.

        Args:
            orchestrator: The active MasterOrchestrator instance.
        """
        self._orchestrator = orchestrator

    def require_refinement_result(self) -> RefinementResult:
        """
        Load and validate the preprocess refinement artifact.

        Returns:
            The parsed RefinementResult from the ArtifactStore.

        Raises:
            ValueError: If the refinement artifact is missing from the cache.
        """
        key = self._key_builder.preprocess_refine_key()
        artifact_data = self._artifact_store.get_artifact(key)

        if not artifact_data:
            raise ValueError(
                f"Missing prerequisite: preprocess refinement artifact (key: {key}). Run 'rrp run-step preprocess refine' first."
            )

        return RefinementResult(**artifact_data)

    def require_assessment_task_list(self, prompt: str) -> AssessmentTaskList:
        """
        Load and validate the assessment decomposition artifact.

        Args:
            prompt: The assessment criteria prompt.

        Returns:
            The parsed AssessmentTaskList from the ArtifactStore.

        Raises:
            ValueError: If the assessment decomposition artifact is missing.
        """
        key = self._key_builder.assessment_decompose_key(prompt)
        artifact_data = self._artifact_store.get_artifact(key)

        if not artifact_data:
            raise ValueError(
                f"Missing prerequisite: assessment decomposition artifact (key: {key}). Run 'run-step assessment decompose' first."
            )

        return AssessmentTaskList(**artifact_data)

    def require_diagnostic_task_list(
        self, prompt: str, assessment_details: list[dict]
    ) -> DiagnosticTaskList:
        """
        Load and validate the diagnostic decomposition artifact.

        Args:
            prompt: The assessment criteria prompt.
            assessment_details: The filtered assessment details targeted for diagnostic.

        Returns:
            The parsed DiagnosticTaskList from the ArtifactStore.

        Raises:
            ValueError: If the diagnostic decomposition artifact is missing.
        """
        key = self._key_builder.diagnostic_decompose_key(prompt, assessment_details)
        artifact_data = self._artifact_store.get_artifact(key)

        if not artifact_data:
            raise ValueError(
                f"Missing prerequisite: diagnostic decomposition artifact (key: {key}). Run 'run-step diagnostic decompose' first."
            )

        return DiagnosticTaskList(**artifact_data)

    def require_assessment_evidence_reports(
        self, task_list: AssessmentTaskList
    ) -> list[AssessmentEvidenceReport]:
        """
        Load all assessment extraction artifacts for a specific task list.

        Args:
            task_list: The execution plan defining the required artifact keys.

        Returns:
            A list of validated AssessmentEvidenceReport instances.

        Raises:
            ValueError: If any group's evidence artifact is missing from the cache.
        """
        evidence_reports = []
        for group in task_list.groups:
            key = self._key_builder.assessment_extract_key(group)
            artifact_data = self._artifact_store.get_artifact(key)

            if not artifact_data:
                raise ValueError(
                    f"Missing prerequisite: assessment extraction artifact for group '{group.group_name}' (key: {key}). "
                    "Run 'run-step assessment extract' first."
                )

            evidence_reports.append(AssessmentEvidenceReport(**artifact_data))

        return evidence_reports

    async def dispatch_assessment_groups(
        self, task_list: AssessmentTaskList, paper_context: PaperContext
    ) -> list[AssessmentAnswer]:
        """
        Coordinate the execution of extraction and synthesis for assessment groups.

        Args:
            task_list: The plan defining the groups to process.
            paper_context: The source document context.

        Returns:
            A flattened list of all AssessmentAnswer objects generated across all groups.
        """

        async def process_group(group: AssessmentGroup) -> list[AssessmentAnswer]:
            evidence = await self._orchestrator.execute_assessment_extraction(group, paper_context)
            report = await self._orchestrator.execute_assessment_synthesis(group, evidence)
            return report.answers

        if self._profile.assessment.extraction.processing_mode == "concurrent":
            logger.info("Running assessment groups concurrently (bounded)...")
            grouped_answers = await self.gather_concurrently(
                [process_group(group) for group in task_list.groups], limit=2
            )
        else:
            logger.info("Running assessment groups sequentially...")
            grouped_answers = []
            for group in task_list.groups:
                grouped_answers.append(await process_group(group))

        return [answer for answer_list in grouped_answers for answer in answer_list]

    async def dispatch_diagnostic_groups(
        self, task_list: DiagnosticTaskList, paper_context: PaperContext, assessment_prompt: str
    ) -> list[DiagnosticItem]:
        """
        Coordinate the execution of analysis for diagnostic groups.

        Args:
            task_list: The plan defining the diagnostic groups.
            paper_context: The source document context.

        Returns:
            A flattened list of all DiagnosticItem objects generated.
        """

        async def process_group(group: DiagnosticGroup) -> list[DiagnosticItem]:
            """Executes diagnostic analysis for a single group."""
            report = await self._orchestrator.execute_diagnostic_analysis(
                group, assessment_prompt, paper_context
            )
            return report.analyses

        if self._profile.diagnostic.analysis.processing_mode == "concurrent":
            logger.info("Running diagnostic groups concurrently (bounded)...")
            grouped_results = await self.gather_concurrently(
                [process_group(group) for group in task_list.groups], limit=2
            )
        else:
            logger.info("Running diagnostic groups sequentially...")
            grouped_results = []
            for group in task_list.groups:
                grouped_results.append(await process_group(group))

        return [item for sub_list in grouped_results for item in sub_list]

    async def gather_concurrently(
        self, coroutines: list[Awaitable[TaskResultType]], limit: int = 2
    ) -> list[TaskResultType]:
        """
        Execute multiple asynchronous tasks with a concurrency limit.

        Uses a semaphore to ensure that no more than 'limit' tasks are in-flight
        simultaneously, preventing API rate limiting or resource exhaustion.

        Args:
            coroutines: The list of tasks to execute.
            limit: The maximum number of concurrent executions.

        Returns:
            A list of results in the same order as the input tasks.
        """
        semaphore = asyncio.Semaphore(limit)

        async def bounded_coroutine(coroutine: Awaitable[TaskResultType]) -> TaskResultType:
            async with semaphore:
                return await coroutine

        return await asyncio.gather(*(bounded_coroutine(coroutine) for coroutine in coroutines))

    def resolve_diagnostic_prompt(self, master_prompt: str) -> str:
        """
        Determine the assessment prompt based on the diagnostic configuration.

        Args:
            master_prompt: The raw assessment criteria.

        Returns:
            The resolved prompt text.
        """
        if (
            self._profile.diagnostic
            and self._profile.diagnostic.prompt_source == DiagnosticPromptSource.REFINED
        ):
            refinement_result = self.require_refinement_result()
            return refinement_result.refined_prompt

        return master_prompt

    def get_identifier_mapping(self) -> dict[str, str | None]:
        """
        Retrieve the semantic-to-original identifier mapping from the refinement phase.

        Returns:
            A dictionary mapping generated identifiers to their original labels.
        """
        refinement_result = self.require_refinement_result()
        identifier_mapping = refinement_result.get_id_map()

        if (
            self._profile.preprocess.refinement.strategy == RefinementStrategy.SEMANTIC
            and not identifier_mapping
        ):
            logger.warning(
                "Refinement strategy is SEMANTIC but no question identifiers found in the refinement artifact."
            )

        return identifier_mapping

    def resolve_original_identifier(
        self, question_identifier: str, identifier_mapping: dict[str, str | None]
    ) -> str:
        """
        Map a generated question identifier back to its source document label.

        Args:
            question_identifier: The identifier generated by the model.
            identifier_mapping: The mapping dictionary from the refinement stage.

        Returns:
            The original label if found, otherwise the input identifier.
        """
        resolved = identifier_mapping.get(question_identifier, question_identifier)
        return resolved if resolved is not None else question_identifier

    def filter_diagnostic_details(
        self,
        assessment_details: list[dict],
        target: DiagnosticAnalysisStrategy,
        identifier_mapping: dict[str, str | None],
        ground_truth: dict[str, bool] | None = None,
    ) -> list[dict]:
        """
        Filter and enrich assessment results for diagnostic analysis.

        Args:
            assessment_details: Raw prediction dictionaries from an AssessmentReport.
            target: The filter criteria (ALL, MISMATCHES, etc.).
            identifier_mapping: The identifier resolution mapping.
            ground_truth: Optional mapping of expected True/False values.

        Returns:
            A list of result dictionaries filtered and annotated with 'expected' and 'correct' flags.
        """
        if not ground_truth:
            if target != DiagnosticAnalysisStrategy.DIAGNOSE_ALL:
                logger.warning(
                    f"No ground truth provided. Falling back to DIAGNOSE_ALL instead of {target.value}."
                )
            return assessment_details

        for assessment_detail in assessment_details:
            question_identifier = assessment_detail["question_id"]
            resolved_identifier = self.resolve_original_identifier(
                question_identifier, identifier_mapping
            )
            expected_answer = ground_truth.get(resolved_identifier)
            is_correct_answer = (
                assessment_detail["answer"] == expected_answer
                if expected_answer is not None
                else None
            )
            assessment_detail["correct"] = is_correct_answer
            assessment_detail["model_answer"] = assessment_detail.pop("answer", None)
            assessment_detail["model_justification"] = assessment_detail.pop("justification", None)
            assessment_detail["ground_truth_answer"] = expected_answer

        if target == DiagnosticAnalysisStrategy.DIAGNOSE_ALL:
            return assessment_details
        elif target == DiagnosticAnalysisStrategy.DIAGNOSE_MISMATCHES:
            return [
                assessment_detail
                for assessment_detail in assessment_details
                if assessment_detail.get("correct") is False
            ]
        elif target == DiagnosticAnalysisStrategy.DIAGNOSE_MATCHES:
            return [
                assessment_detail
                for assessment_detail in assessment_details
                if assessment_detail.get("correct") is True
            ]
        elif target == DiagnosticAnalysisStrategy.DIAGNOSE_OVERPREDICTIONS:
            return [
                assessment_detail
                for assessment_detail in assessment_details
                if assessment_detail["model_answer"] is True
                and assessment_detail["ground_truth_answer"] is False
            ]
        elif target == DiagnosticAnalysisStrategy.DIAGNOSE_UNDERPREDICTIONS:
            return [
                assessment_detail
                for assessment_detail in assessment_details
                if assessment_detail["model_answer"] is False
                and assessment_detail["ground_truth_answer"] is True
            ]

        return assessment_details

    def get_assessment_artifacts_for_diagnostic(self, prompt: str) -> dict[str, list[dict]]:
        """
        Assemble the full context from the assessment phase for diagnostic investigation.

        Loads all evidence fragments collected during the assessment stage.

        Args:
            prompt: The assessment criteria prompt used to generate the assessment task list.

        Returns:
            A dictionary containing the evidence fragments.
        """
        assessment_evidence = []
        task_list_data = self._artifact_store.get_artifact(
            self._key_builder.assessment_decompose_key(prompt)
        )
        if task_list_data:
            task_list = AssessmentTaskList(**task_list_data)
            for group in task_list.groups:
                cached_evidence = self._artifact_store.get_artifact(
                    self._key_builder.assessment_extract_key(group)
                )
                if cached_evidence:
                    assessment_evidence.append(cached_evidence)

        return {"evidence": assessment_evidence}
