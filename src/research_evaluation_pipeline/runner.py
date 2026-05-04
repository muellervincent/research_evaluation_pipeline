"""
Pipeline Runner Service.

Provides a high-level API to execute the research evaluation pipeline.
Stateless and path-agnostic, accepting pre-loaded data.
"""

import hashlib
from pathlib import Path

from loguru import logger

from .clients.factory import ProviderFactory
from .config.client_settings import ClientProfile
from .config.execution_settings import PipelineProfile
from .core.artifact_store import ArtifactStore
from .core.enums import FragmentationMode, IngestionMode
from .core.master_orchestrator import MasterOrchestrator
from .core.paper_context import PaperContext
from .core.step_executor import StepExecutor
from .result.builder import ResultBuilder
from .service.artifact_key_builder import ArtifactKeyBuilder
from .service.paper_context_service import PaperContextService
from .service.prompt_service import PromptService


class RunnerError(Exception):
    """Custom exception for errors occurring within the PipelineRunner."""


class PipelineRunner:
    """
    High-level API for running the research evaluation pipeline.

    This class is stateless and path-agnostic, accepting pre-loaded
    data and profiles to execute the pipeline stages.
    """

    def __init__(self):
        """
        Initialize the runner.
        """
        pass

    @staticmethod
    def generate_prompt_key(prompt_text: str) -> str:
        """
        Generate a deterministic 8-character hash for a prompt.
        """
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8]

    def setup_orchestrator(
        self,
        profile: PipelineProfile,
        client_profile: ClientProfile,
        paper_stem: str,
        master_prompt_key: str,
        artifact_store_path: Path = Path("resources/artifacts.db"),
    ) -> MasterOrchestrator:
        """
        Construct the full dependency graph for the orchestrator.
        """
        artifact_store = ArtifactStore(database_path=artifact_store_path)
        prompt_service = PromptService()
        provider = ProviderFactory.get_provider(client_profile)

        key_builder = ArtifactKeyBuilder(profile, paper_stem, master_prompt_key)
        paper_context_service = PaperContextService(
            provider=provider,
            artifact_store=artifact_store,
            key_builder=key_builder,
            profile=profile,
        )
        step_executor = StepExecutor(
            artifact_store=artifact_store, key_builder=key_builder, profile=profile
        )

        return MasterOrchestrator(
            provider=provider,
            profile=profile,
            prompt_service=prompt_service,
            artifact_store=artifact_store,
            key_builder=key_builder,
            paper_context_service=paper_context_service,
            step_executor=step_executor,
        )

    async def run_pipeline(
        self,
        paper_stem: str,
        paper_bytes: bytes,
        master_prompt: str,
        profile: PipelineProfile,
        client_profile: ClientProfile,
        ground_truth: dict[str, bool] | None = None,
    ):
        """
        Execute the entire research assessment and diagnostic pipeline end-to-end.
        """
        master_prompt_key = self.generate_prompt_key(master_prompt)

        orchestrator = self.setup_orchestrator(
            profile, client_profile, paper_stem, master_prompt_key
        )

        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_stem=paper_stem, raw_bytes=paper_bytes
            )
            await self._run_preprocess_stage(orchestrator, master_prompt, paper_context)
            await self._run_assessment_stage(orchestrator, paper_context)

            assessment_prompt = orchestrator.step_executor.resolve_diagnostic_prompt(master_prompt)

            await self._run_diagnostic_stage(
                orchestrator, assessment_prompt, paper_context, ground_truth
            )
            await self._run_results_stage(orchestrator, master_prompt, ground_truth)
            logger.success("Pipeline complete.")
        finally:
            if paper_context:
                await orchestrator.provider.cleanup_context(paper_context)

    async def run_stage(
        self,
        stage: str,
        paper_stem: str,
        paper_bytes: bytes,
        master_prompt: str,
        profile: PipelineProfile,
        client_profile: ClientProfile,
        ground_truth: dict[str, bool] | None = None,
    ):
        """
        Execute a single stage of the research pipeline.
        """
        master_prompt_key = self.generate_prompt_key(master_prompt)

        orchestrator = self.setup_orchestrator(
            profile, client_profile, paper_stem, master_prompt_key
        )

        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_stem=paper_stem, raw_bytes=paper_bytes
            )

            if stage == "preprocess":
                await self._run_preprocess_stage(orchestrator, master_prompt, paper_context)
            elif stage == "assessment":
                await self._run_assessment_stage(orchestrator, paper_context)
            elif stage == "diagnostic":
                assessment_prompt = orchestrator.step_executor.resolve_diagnostic_prompt(
                    master_prompt
                )
                await self._run_diagnostic_stage(
                    orchestrator, assessment_prompt, paper_context, ground_truth
                )
            elif stage == "result":
                await self._run_results_stage(orchestrator, master_prompt, ground_truth)
            else:
                raise RunnerError(f"Stage '{stage}' is not supported.")

        finally:
            if paper_context:
                await orchestrator.provider.cleanup_context(paper_context)

    async def run_step(
        self,
        stage: str,
        step: str,
        paper_stem: str,
        paper_bytes: bytes,
        master_prompt: str,
        profile: PipelineProfile,
        client_profile: ClientProfile,
        ground_truth: dict[str, bool] | None = None,
    ):
        """
        Execute a granular atomic step with strict prerequisite validation.
        """
        master_prompt_key = self.generate_prompt_key(master_prompt)

        orchestrator = self.setup_orchestrator(
            profile, client_profile, paper_stem, master_prompt_key
        )

        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_stem=paper_stem, raw_bytes=paper_bytes
            )

            if stage == "preprocess":
                await self._run_preprocess_step(orchestrator, step, master_prompt, paper_context)

            elif stage == "assessment":
                await self._run_assessment_step(orchestrator, step, paper_context)

            elif stage == "diagnostic":
                assessment_prompt = orchestrator.step_executor.resolve_diagnostic_prompt(
                    master_prompt
                )
                await self._run_diagnostic_step(
                    orchestrator, step, assessment_prompt, paper_context, ground_truth
                )

            else:
                raise RunnerError(f"Stage '{stage}' is not supported.")

        finally:
            if paper_context:
                await orchestrator.provider.cleanup_context(paper_context)

    async def _run_preprocess_step(
        self,
        orchestrator: MasterOrchestrator,
        step: str,
        master_prompt: str,
        paper_context: PaperContext,
    ):
        """
        Internal dispatcher for preprocess-specific steps.
        """
        if step == "refine":
            await orchestrator.execute_preprocess_refine(master_prompt)
            logger.success("Preprocess refinement complete.")

        elif step == "extract":
            await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
            await orchestrator.execute_preprocess_extraction(paper_context)
            await orchestrator.paper_context_service.restore_extracted_markdown(paper_context)
            logger.success("Preprocess extraction complete.")

        else:
            raise RunnerError(f"Step '{step}' is not valid for the preprocess stage.")

    async def _run_preprocess_stage(
        self, orchestrator: MasterOrchestrator, master_prompt: str, paper_context: PaperContext
    ):
        """
        Execute the full preprocess stage.
        """
        await self._run_preprocess_step(orchestrator, "refine", master_prompt, paper_context)
        if orchestrator.profile.ingestion_mode == IngestionMode.MD:
            await self._run_preprocess_step(orchestrator, "extract", master_prompt, paper_context)

    async def _run_assessment_step(
        self, orchestrator: MasterOrchestrator, step: str, paper_context: PaperContext
    ):
        """
        Internal dispatcher for assessment-specific steps.
        """
        if step == "fast":
            refinement_result = orchestrator.step_executor.require_refinement_result()
            await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
            await orchestrator.paper_context_service.ensure_api_cache(
                paper_context, orchestrator.profile.assessment.synthesis.model.value
            )
            await orchestrator.execute_fast_assessment(
                paper_context, refinement_result.refined_prompt
            )
            logger.success("Assessment FAST mode complete.")

        elif step == "decompose":
            refinement_result = orchestrator.step_executor.require_refinement_result()
            await orchestrator.execute_assessment_decomposition(refinement_result.refined_prompt)
            logger.success("Assessment decomposition complete.")

        elif step == "extract":
            refinement_result = orchestrator.step_executor.require_refinement_result()
            task_list = orchestrator.step_executor.require_assessment_task_list(
                refinement_result.refined_prompt
            )
            await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
            await orchestrator.paper_context_service.ensure_api_cache(
                paper_context, orchestrator.profile.assessment.extraction.model.value
            )
            await orchestrator.step_executor.dispatch_assessment_groups(task_list, paper_context)
            logger.success("Assessment extraction complete for all groups.")

        elif step == "synthesize":
            refinement_result = orchestrator.step_executor.require_refinement_result()
            task_list = orchestrator.step_executor.require_assessment_task_list(
                refinement_result.refined_prompt
            )
            evidence_reports = orchestrator.step_executor.require_assessment_evidence_reports(
                task_list
            )
            for group, evidence in zip(task_list.groups, evidence_reports):
                await orchestrator.execute_assessment_synthesis(group, evidence)
            logger.success("Assessment synthesis complete for all groups.")

        else:
            raise RunnerError(f"Step '{step}' is not valid for the assessment stage.")

    async def _run_assessment_stage(
        self, orchestrator: MasterOrchestrator, paper_context: PaperContext
    ):
        """
        Execute the full assessment stage.
        """
        if orchestrator.profile.assessment.fragmentation == FragmentationMode.FAST:
            await self._run_assessment_step(orchestrator, "fast", paper_context)
        else:
            await self._run_assessment_step(orchestrator, "decompose", paper_context)
            await self._run_assessment_step(orchestrator, "extract", paper_context)
            await self._run_assessment_step(orchestrator, "synthesize", paper_context)

    async def _run_diagnostic_step(
        self,
        orchestrator: MasterOrchestrator,
        step: str,
        assessment_prompt: str,
        paper_context: PaperContext,
        ground_truth: dict[str, bool] | None,
    ):
        """
        Internal dispatcher for diagnostic-specific steps.
        """
        if not orchestrator.profile.diagnostic:
            raise RunnerError("No diagnostic profile defined in the active recipe.")

        assessment_report = await orchestrator.reconstruct_assessment_report()
        if not assessment_report:
            raise RunnerError("Missing Assessment artifacts. Run 'assessment' stage first.")

        identifier_mapping = orchestrator.step_executor.get_identifier_mapping()
        assessment_details = [answer.model_dump() for answer in assessment_report.answers]

        strategy_target = orchestrator.profile.diagnostic.analysis.strategy
        filtered_assessment_details = orchestrator.step_executor.filter_diagnostic_details(
            assessment_details, strategy_target, identifier_mapping, ground_truth
        )

        if not filtered_assessment_details:
            logger.info(
                "No assessment_details matched the diagnostic target criteria. Cannot proceed."
            )
            return

        if step == "fast":
            await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
            await orchestrator.paper_context_service.ensure_api_cache(
                paper_context, orchestrator.profile.diagnostic.analysis.model.value
            )
            await orchestrator.execute_fast_diagnostic(
                filtered_assessment_details, assessment_prompt, paper_context
            )
            logger.success("Diagnostic FAST mode complete.")

        elif step == "decompose":
            await orchestrator.execute_diagnostic_decomposition(
                filtered_assessment_details, assessment_prompt
            )
            logger.success("Diagnostic decomposition complete.")

        elif step == "analyze":
            task_list = orchestrator.step_executor.require_diagnostic_task_list(
                assessment_prompt, filtered_assessment_details
            )
            await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
            await orchestrator.paper_context_service.ensure_api_cache(
                paper_context, orchestrator.profile.diagnostic.analysis.model.value
            )
            await orchestrator.step_executor.dispatch_diagnostic_groups(
                task_list, paper_context, assessment_prompt
            )
            logger.success("Diagnostic analysis complete for all groups.")

        else:
            raise RunnerError(f"Step '{step}' is not valid for the diagnostic stage.")

    async def _run_diagnostic_stage(
        self,
        orchestrator: MasterOrchestrator,
        assessment_prompt: str,
        paper_context: PaperContext,
        ground_truth: dict[str, bool] | None,
    ):
        """
        Execute the full diagnostic stage.
        """
        if not orchestrator.profile.diagnostic:
            return
        if orchestrator.profile.diagnostic.fragmentation == FragmentationMode.FAST:
            await self._run_diagnostic_step(
                orchestrator, "fast", assessment_prompt, paper_context, ground_truth
            )
        else:
            await self._run_diagnostic_step(
                orchestrator, "decompose", assessment_prompt, paper_context, ground_truth
            )
            await self._run_diagnostic_step(
                orchestrator, "analyze", assessment_prompt, paper_context, ground_truth
            )

    async def _run_results_stage(
        self,
        orchestrator: MasterOrchestrator,
        master_prompt: str,
        ground_truth: dict[str, bool] | None,
    ):
        """
        Execute the final result generation stage.
        """

        logger.info("Starting result generation...")

        refinement_result = orchestrator.step_executor.require_refinement_result()
        prompt = refinement_result.refined_prompt

        assessment_report = await orchestrator.reconstruct_assessment_report()
        if not assessment_report:
            logger.error("Assessment report is missing. Cannot generate final results.")
            return

        identifier_mapping = orchestrator.step_executor.get_identifier_mapping()

        diagnostic_report = None
        if orchestrator.profile.diagnostic:
            assessment_details = [answer.model_dump() for answer in assessment_report.answers]
            filtered_assessment_details = orchestrator.step_executor.filter_diagnostic_details(
                assessment_details,
                orchestrator.profile.diagnostic.analysis.strategy,
                identifier_mapping,
                ground_truth,
            )
            diagnostic_report = await orchestrator.reconstruct_diagnostic_report(
                prompt, filtered_assessment_details
            )
            if not diagnostic_report:
                logger.warning("Diagnostic profile is enabled but diagnostic report is missing.")

        final_result = ResultBuilder.build_final_result(
            profile=orchestrator.profile,
            assessment_report=assessment_report,
            identifier_mapping=identifier_mapping,
            ground_truth=ground_truth,
            paper_stem=orchestrator.key_builder.paper_stem,
            master_prompt_key=orchestrator.key_builder.master_prompt_key,
            diagnostic_report=diagnostic_report,
            refined_prompt=prompt,
        )

        markdown_report = ResultBuilder.build_markdown_report(final_result, orchestrator.profile)
        settings_hex = ResultBuilder.get_settings_hex(orchestrator.profile)

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        base_filename = f"{orchestrator.profile.ingestion_mode.value}_{orchestrator.key_builder.paper_stem}_{orchestrator.key_builder.master_prompt_key}_{settings_hex}"

        json_path = output_dir / f"{base_filename}.json"
        with open(json_path, "w") as result_file:
            result_file.write(final_result.model_dump_json(indent=2))

        md_path = output_dir / f"{base_filename}.md"
        with open(md_path, "w") as report_file:
            report_file.write(markdown_report)

        logger.success(f"Generated results saved to {output_dir}/")
