"""
Pipeline Runner Service.

Provides a high-level API to execute the research evaluation pipeline,
independent of the CLI or frontend.
"""

import csv
import hashlib
import json
from pathlib import Path

import tomli
import yaml
from loguru import logger

from .clients.factory import ProviderFactory
from .config.client_settings import ClientProfile
from .config.execution_settings import PipelineProfile
from .config.prompt_registry import PromptRegistry
from .core.artifact_store import ArtifactStore
from .core.domain import PaperContext
from .core.enums import FragmentationMode, IngestionMode
from .service.artifact_key_builder import ArtifactKeyBuilder
from .service.master_orchestrator import MasterOrchestrator
from .service.paper_context_service import PaperContextService
from .service.step_executor import StepExecutor
from .reporting.generator import ResultGenerator


class RunnerError(Exception):
    """Custom exception for errors occurring within the PipelineRunner."""


class PipelineRunner:
    """
    High-level API for running the research evaluation pipeline.

    This class encapsulates input resolution, profile management, orchestrator
    setup, and the execution logic for stages and steps.
    """

    def __init__(
        self,
        execution_profiles_path: Path = Path("resources/profiles/execution.toml"),
        client_profiles_path: Path = Path("resources/profiles/client.toml"),
    ):
        self.execution_profiles_path = execution_profiles_path
        self.client_profiles_path = client_profiles_path

    def resolve_inputs(
        self,
        paper_path: Path,
        prompt_path: Path,
        prompt_key: str | None,
        ground_truth_path: Path | None,
    ) -> tuple[str, str, str]:
        """
        Resolve and validate all execution inputs.

        Handles:
        - Deriving paper stem from PDF filename.
        - Loading prompt content from MD/TXT or JSON/YAML.
        - Generating a deterministic master_prompt_key via hashing.

        Returns:
            tuple of (paper_stem, master_prompt_text, master_prompt_key)

        Raises:
            RunnerError: If inputs are invalid or missing.
        """

        if not paper_path.exists():
            raise RunnerError(f"Paper not found: {paper_path}")
        paper_stem = paper_path.stem

        if not prompt_path.exists():
            raise RunnerError(f"Prompt file not found: {prompt_path}")

        extension = prompt_path.suffix.lower()
        master_prompt_text = ""

        if extension in (".md", ".txt"):
            master_prompt_text = prompt_path.read_text(encoding="utf-8")
        elif extension in (".json", ".yaml", ".yml"):
            with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                data = (
                    json.load(prompt_file) if extension == ".json" else yaml.safe_load(prompt_file)
                )

            if prompt_key:
                if prompt_key not in data:
                    raise RunnerError(f"Key '{prompt_key}' not found in {prompt_path}")
                master_prompt_text = data[prompt_key]
            else:
                if isinstance(data, str):
                    master_prompt_text = data
                elif isinstance(data, dict) and len(data) == 1:
                    master_prompt_text = list(data.values())[0]
                else:
                    raise RunnerError("Prompt key is required for registry files.")
        else:
            raise RunnerError(f"Unsupported prompt file extension: {extension}")

        master_prompt_key = hashlib.sha256(master_prompt_text.encode("utf-8")).hexdigest()[:8]

        if not ground_truth_path or not ground_truth_path.exists():
            raise RunnerError("Ground truth file not found.")

        return paper_stem, master_prompt_text, master_prompt_key

    def load_profile(self, profile_name: str) -> PipelineProfile:
        """
        Load and validate a PipelineProfile from the filesystem.
        """
        if not self.execution_profiles_path.exists():
            raise RunnerError(f"Profiles file not found: {self.execution_profiles_path}")

        with open(self.execution_profiles_path, "rb") as profile_file:
            profiles_data = tomli.load(profile_file)

        if profile_name not in profiles_data:
            raise RunnerError(
                f"Profile '{profile_name}' not found in {self.execution_profiles_path}"
            )

        try:
            return PipelineProfile(**profiles_data[profile_name])
        except Exception as error:
            raise RunnerError(f"Failed to validate PipelineProfile '{profile_name}': {error}")

    def load_client_profile(self, profile_name: str) -> ClientProfile:
        """
        Load and validate a ClientProfile from the filesystem.
        """
        if not self.client_profiles_path.exists():
            raise RunnerError(f"Client profiles file not found: {self.client_profiles_path}")

        with open(self.client_profiles_path, "rb") as profile_file:
            profiles_data = tomli.load(profile_file)

        if profile_name not in profiles_data:
            raise RunnerError(
                f"Client profile '{profile_name}' not found in {self.client_profiles_path}"
            )

        try:
            return ClientProfile(**profiles_data[profile_name])
        except Exception as error:
            raise RunnerError(f"Failed to validate ClientProfile '{profile_name}': {error}")

    def setup_orchestrator(
        self, profile_name: str, client_profile_name: str, paper_stem: str, master_prompt_key: str
    ) -> MasterOrchestrator:
        """
        Construct the full dependency graph for the orchestrator.
        """
        profile = self.load_profile(profile_name)
        client_profile = self.load_client_profile(client_profile_name)

        artifact_store = ArtifactStore()
        prompt_registry = PromptRegistry()
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
            prompt_registry=prompt_registry,
            artifact_store=artifact_store,
            key_builder=key_builder,
            paper_context_service=paper_context_service,
            step_executor=step_executor,
            paper_stem=paper_stem,
            master_prompt_key=master_prompt_key,
        )

    def load_ground_truth(
        self, ground_truth_path: Path | None, paper_stem: str
    ) -> dict[str, bool] | None:
        """
        Retrieve expected answers for a specific paper from a CSV file.
        """
        if not ground_truth_path or not ground_truth_path.exists():
            return None
        try:
            study_identifier = paper_stem.lstrip("0")
            ground_truth_data = {}
            with open(ground_truth_path, "r") as ground_truth_file:
                reader = csv.DictReader(ground_truth_file, delimiter=";")
                for row in reader:
                    if row["study_number"] == study_identifier:
                        answer_text = row["answer"].strip()
                        prompt_number = row["prompt_number"].strip()
                        answer_value = None
                        if answer_text == "1":
                            answer_value = True
                        elif answer_text == "0":
                            answer_value = False

                        if answer_value is not None:
                            ground_truth_data[prompt_number] = answer_value
                            ground_truth_data[f"{prompt_number}."] = answer_value
            return ground_truth_data if ground_truth_data else None
        except Exception as exception:
            logger.error(f"Failed to load ground truth from {ground_truth_path}: {exception}")
            return None

    async def run_pipeline(
        self,
        paper_path: Path,
        prompt_path: Path,
        ground_truth_path: Path,
        profile_name: str = "standard",
        client_profile_name: str = "unpaid",
        prompt_key: str | None = None,
    ):
        """
        Execute the entire research assessment and diagnostic pipeline end-to-end.
        """
        paper_stem, master_prompt, master_prompt_key = self.resolve_inputs(
            paper_path, prompt_path, prompt_key, ground_truth_path
        )

        orchestrator = self.setup_orchestrator(
            profile_name, client_profile_name, paper_stem, master_prompt_key
        )

        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_path
            )
            await self._run_preprocess_stage(orchestrator, master_prompt, paper_context)
            await self._run_assessment_stage(orchestrator, paper_context)

            if (
                orchestrator.profile.diagnostic
                and orchestrator.profile.diagnostic.prompt_source.value == "refined"
            ):
                refinement_result = orchestrator.step_executor.require_refinement_result()
                assessment_prompt = refinement_result.refined_prompt
            else:
                assessment_prompt = master_prompt

            await self._run_diagnostic_stage(
                orchestrator, assessment_prompt, paper_context, ground_truth_path
            )
            await self._run_results_stage(
                orchestrator, profile_name, master_prompt, ground_truth_path
            )
            logger.success("Pipeline complete.")
        finally:
            if paper_context:
                await orchestrator.provider.cleanup_context(paper_context)

    async def run_stage(
        self,
        stage: str,
        paper_path: Path,
        prompt_path: Path,
        ground_truth_path: Path,
        profile_name: str = "standard",
        client_profile_name: str = "unpaid",
        prompt_key: str | None = None,
    ):
        """
        Execute a single stage of the research pipeline.
        """
        paper_stem, master_prompt, master_prompt_key = self.resolve_inputs(
            paper_path, prompt_path, prompt_key, ground_truth_path
        )

        orchestrator = self.setup_orchestrator(
            profile_name, client_profile_name, paper_stem, master_prompt_key
        )

        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_path
            )

            if stage == "preprocess":
                await self._run_preprocess_stage(orchestrator, master_prompt, paper_context)
            elif stage == "assessment":
                await self._run_assessment_stage(orchestrator, paper_context)
            elif stage == "diagnostic":
                if (
                    orchestrator.profile.diagnostic
                    and orchestrator.profile.diagnostic.prompt_source.value == "refined"
                ):
                    refinement_result = orchestrator.step_executor.require_refinement_result()
                    assessment_prompt = refinement_result.refined_prompt
                else:
                    assessment_prompt = master_prompt
                await self._run_diagnostic_stage(
                    orchestrator, assessment_prompt, paper_context, ground_truth_path
                )
            elif stage == "results":
                await self._run_results_stage(
                    orchestrator, profile_name, master_prompt, ground_truth_path
                )
            else:
                raise RunnerError(f"Stage '{stage}' is not supported.")

        finally:
            if paper_context:
                await orchestrator.provider.cleanup_context(paper_context)

    async def run_step(
        self,
        stage: str,
        step: str,
        paper_path: Path,
        prompt_path: Path,
        ground_truth_path: Path,
        profile_name: str = "standard",
        client_profile_name: str = "unpaid",
        prompt_key: str | None = None,
    ):
        """
        Execute a granular atomic step with strict prerequisite validation.
        """
        paper_stem, master_prompt, master_prompt_key = self.resolve_inputs(
            paper_path, prompt_path, prompt_key, ground_truth_path
        )

        orchestrator = self.setup_orchestrator(
            profile_name, client_profile_name, paper_stem, master_prompt_key
        )

        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_path
            )

            if stage == "preprocess":
                await self._run_preprocess_step(orchestrator, step, master_prompt, paper_context)

            elif stage == "assessment":
                await self._run_assessment_step(orchestrator, step, paper_context)

            elif stage == "diagnostic":
                if (
                    orchestrator.profile.diagnostic
                    and orchestrator.profile.diagnostic.prompt_source.value == "refined"
                ):
                    refinement_result = orchestrator.step_executor.require_refinement_result()
                    assessment_prompt = refinement_result.refined_prompt
                else:
                    assessment_prompt = master_prompt
                await self._run_diagnostic_step(
                    orchestrator, step, assessment_prompt, paper_context, ground_truth_path
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
        if paper_context.ingestion_mode == IngestionMode.EXTRACTION:
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
            await orchestrator.paper_context_service.ensure_application_programming_interface_cache(
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
            await orchestrator.paper_context_service.ensure_application_programming_interface_cache(
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
        ground_truth_path: Path | None,
    ):
        """
        Internal dispatcher for diagnostic-specific steps.
        """
        if not orchestrator.profile.diagnostic:
            raise RunnerError("No diagnostic profile defined in the active recipe.")

        assessment_report = await orchestrator.reconstruct_assessment_report()
        if not assessment_report:
            raise RunnerError("Missing Assessment artifacts. Run 'assessment' stage first.")

        ground_truth_data = self.load_ground_truth(ground_truth_path, orchestrator.paper_stem)
        identifier_mapping = orchestrator.step_executor.get_identifier_mapping()
        assessment_details = [answer.model_dump() for answer in assessment_report.answers]

        strategy_target = orchestrator.profile.diagnostic.analysis.strategy
        filtered_assessment_details = orchestrator.step_executor.filter_diagnostic_details(
            assessment_details, strategy_target, identifier_mapping, ground_truth_data
        )

        if not filtered_assessment_details:
            logger.info(
                "No assessment_details matched the diagnostic target criteria. Cannot proceed."
            )
            return

        if step == "fast":
            await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
            await orchestrator.paper_context_service.ensure_application_programming_interface_cache(
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
            await orchestrator.paper_context_service.ensure_application_programming_interface_cache(
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
        ground_truth_path: Path | None,
    ):
        """
        Execute the full diagnostic stage.
        """
        if not orchestrator.profile.diagnostic:
            return
        if orchestrator.profile.diagnostic.fragmentation == FragmentationMode.FAST:
            await self._run_diagnostic_step(
                orchestrator, "fast", assessment_prompt, paper_context, ground_truth_path
            )
        else:
            await self._run_diagnostic_step(
                orchestrator, "decompose", assessment_prompt, paper_context, ground_truth_path
            )
            await self._run_diagnostic_step(
                orchestrator, "analyze", assessment_prompt, paper_context, ground_truth_path
            )

    async def _run_results_stage(
        self,
        orchestrator: MasterOrchestrator,
        profile_name: str,
        master_prompt: str,
        resolved_ground_truth_path: Path | None,
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
        ground_truth_data = self.load_ground_truth(
            resolved_ground_truth_path, orchestrator.paper_stem
        )

        diagnostic_report = None
        if orchestrator.profile.diagnostic:
            assessment_details = [answer.model_dump() for answer in assessment_report.answers]
            filtered_assessment_details = orchestrator.step_executor.filter_diagnostic_details(
                assessment_details,
                orchestrator.profile.diagnostic.analysis.strategy,
                identifier_mapping,
                ground_truth_data,
            )
            diagnostic_report = await orchestrator.reconstruct_diagnostic_report(
                prompt, filtered_assessment_details
            )
            if not diagnostic_report:
                logger.warning("Diagnostic profile is enabled but diagnostic report is missing.")

        final_result = ResultGenerator.build_final_result(
            profile=orchestrator.profile,
            assessment_report=assessment_report,
            identifier_mapping=identifier_mapping,
            ground_truth=ground_truth_data,
            paper_stem=orchestrator.paper_stem,
            master_prompt_key=orchestrator.master_prompt_key,
            diagnostic_report=diagnostic_report,
            refined_prompt=prompt,
        )

        markdown_report = ResultGenerator.build_markdown_report(final_result, orchestrator.profile)
        settings_hex = ResultGenerator.get_settings_hex(orchestrator.profile)

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        base_filename = f"{profile_name}_{orchestrator.paper_stem}_{orchestrator.master_prompt_key}_{settings_hex}"

        json_path = output_dir / f"{base_filename}.json"
        with open(json_path, "w") as result_file:
            result_file.write(final_result.model_dump_json(indent=2))

        md_path = output_dir / f"{base_filename}.md"
        with open(md_path, "w") as report_file:
            report_file.write(markdown_report)

        logger.success(f"Generated results saved to {output_dir}/")
