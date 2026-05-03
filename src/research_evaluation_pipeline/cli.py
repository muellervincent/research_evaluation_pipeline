"""
Typer CLI for the Research Pipeline.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import tomli
import typer
import yaml
from loguru import logger

from .clients.factory import ProviderFactory
from .config.client_settings import ClientProfile
from .config.execution_settings import PipelineProfile
from .config.prompt_registry import PromptRegistry
from .core.enums import PipelineStage
from .core.artifact_store import ArtifactStore
from .reporting.generator import ResultGenerator
from .service.artifact_key_builder import ArtifactKeyBuilder
from .service.master_orchestrator import MasterOrchestrator
from .service.paper_context_service import PaperContextService
from .service.step_executor import StepExecutor

EXECUTION_PROFILES_PATH = Path("resources/profiles/execution.toml")
CLIENT_PROFILES_PATH = Path("resources/profiles/client.toml")

app = typer.Typer(help="Research Reporting Pipeline CLI", no_args_is_help=True)
database_app = typer.Typer(help="Database management commands", no_args_is_help=True)
app.add_typer(database_app, name="db")


def _resolve_inputs(
    paper_path: Path, prompt_path: Path, prompt_key: str | None, ground_truth_path: Path | None
) -> tuple[str, str, str]:
    """
    Resolve and validate all execution inputs.

    Handles:
    - Deriving paper stem from PDF filename.
    - Loading prompt content from MD/TXT or JSON/YAML.
    - Generating a deterministic master_prompt_key via hashing.

    Returns:
        tuple of (paper_stem, master_prompt_text, master_prompt_key)
    """

    if not paper_path.exists():
        logger.error(f"Paper not found: {paper_path}")
        raise typer.Exit(1)
    paper_stem = paper_path.stem

    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        raise typer.Exit(1)

    extension = prompt_path.suffix.lower()
    master_prompt_text = ""

    if extension in (".md", ".txt"):
        master_prompt_text = prompt_path.read_text(encoding="utf-8")
    elif extension in (".json", ".yaml", ".yml"):
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            data = json.load(prompt_file) if extension == ".json" else yaml.safe_load(prompt_file)

        if prompt_key:
            if prompt_key not in data:
                logger.error(f"Key '{prompt_key}' not found in {prompt_path}")
                raise typer.Exit(1)
            master_prompt_text = data[prompt_key]
        else:
            if isinstance(data, str):
                master_prompt_text = data
            elif isinstance(data, dict) and len(data) == 1:
                master_prompt_text = list(data.values())[0]
            else:
                logger.error("Prompt key (--prompt-key) is required for registry files.")
                raise typer.Exit(1)
    else:
        logger.error(f"Unsupported prompt file extension: {extension}")
        raise typer.Exit(1)

    master_prompt_key = hashlib.sha256(master_prompt_text.encode("utf-8")).hexdigest()[:8]

    if not ground_truth_path or not ground_truth_path.exists():
        logger.error("Ground truth file not found.")
        raise typer.Exit(1)

    return paper_stem, master_prompt_text, master_prompt_key


def _load_profile(profile_name: str, profile_path: Path) -> PipelineProfile:
    """
    Load and validate a PipelineProfile from the filesystem.

    Args:
        profile_name: The name of the profile section.
        profile_path: The filesystem path to the TOML profile file.

    Returns:
        A validated PipelineProfile instance.

    Raises:
        typer.Exit: If the profile is missing or invalid.
    """

    if not profile_path.exists():
        logger.error(f"Profiles file not found: {profile_path}")
        raise typer.Exit(1)

    with open(profile_path, "rb") as profile_file:
        profiles_data = tomli.load(profile_file)

    if profile_name not in profiles_data:
        logger.error(f"Profile '{profile_name}' not found in {profile_path}")
        raise typer.Exit(1)

    profile_dict = profiles_data[profile_name]

    try:
        return PipelineProfile(**profile_dict)
    except Exception as error:
        logger.error(f"Failed to validate PipelineProfile '{profile_name}': {error}")
        raise typer.Exit(1)


def _load_client_profile(profile_name: str, profile_path: Path) -> ClientProfile:
    """
    Load and validate a ClientProfile from the filesystem.

    Args:
        profile_name: The name of the profile section.
        profile_path: The filesystem path to the TOML client profile file.

    Returns:
        A validated ClientProfile instance.

    Raises:
        typer.Exit: If the profile is missing or invalid.
    """

    if not profile_path.exists():
        logger.error(f"Client profiles file not found: {profile_path}")
        raise typer.Exit(1)

    with open(profile_path, "rb") as profile_file:
        profiles_data = tomli.load(profile_file)

    if profile_name not in profiles_data:
        logger.error(f"Client profile '{profile_name}' not found in {profile_path}")
        raise typer.Exit(1)

    profile_dict = profiles_data[profile_name]

    try:
        return ClientProfile(**profile_dict)
    except Exception as error:
        logger.error(f"Failed to validate ClientProfile '{profile_name}': {error}")
        raise typer.Exit(1)


def _setup_orchestrator(
    profile_name: str,
    execution_profiles_path: Path,
    client_profile_name: str,
    client_profiles_path: Path,
    paper_stem: str,
    master_prompt_key: str,
) -> MasterOrchestrator:
    """
    Construct the full dependency graph for the orchestrator.

    Args:
        profile_name: The name of the strategy profile.
        execution_profiles_path: Path to the execution profiles TOML.
        client_profile_name: The name of the client profile.
        client_profiles_path: Path to the client profiles TOML.
        paper_stem: The identifier for the paper.
        master_prompt_key: The identifier for the prompt.

    Returns:
        A MasterOrchestrator instance.
    """
    profile = _load_profile(profile_name, execution_profiles_path)
    client_profile = _load_client_profile(client_profile_name, client_profiles_path)

    artifact_store = ArtifactStore()
    prompt_registry = PromptRegistry()
    provider = ProviderFactory.get_provider(client_profile)

    key_builder = ArtifactKeyBuilder(profile, paper_stem, master_prompt_key)
    paper_context_service = PaperContextService(
        provider=provider, artifact_store=artifact_store, key_builder=key_builder, profile=profile
    )
    step_executor = StepExecutor(
        artifact_store=artifact_store, key_builder=key_builder, profile=profile
    )

    orchestrator = MasterOrchestrator(
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

    return orchestrator


def _load_ground_truth_from_path(
    ground_truth_path: Path | None, paper_stem: str
) -> dict[str, bool] | None:
    """
    Retrieve expected answers for a specific paper from a CSV file.

    Args:
        ground_truth_path: The path to the ground truth CSV.
        paper_stem: The identifier of the paper.

    Returns:
        A dictionary mapping question labels to boolean answers, or None if not found.
    """
    if not ground_truth_path or not ground_truth_path.exists():
        return None
    try:
        import csv

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


@app.command()
def run_pipeline(
    profile_name: str = typer.Option("standard", "--profile", help="Name of the strategy profile to use"),
    client_profile: str = typer.Option(
        "unpaid", "--client-profile", help="Name of the client profile to use"
    ),
    paper_path: Path = typer.Option(..., "--paper-path", help="Path to the source PDF"),
    prompt_path: Path = typer.Option(..., "--prompt-path", help="Path to the criteria file"),
    prompt_key: str | None = typer.Option(
        None, "--prompt-key", help="Key if using a registry file"
    ),
    ground_truth_path: Path = typer.Option(
        ..., "--ground-truth-path", help="Path to ground truth CSV"
    ),
    execution_profiles: Path = typer.Option(
        EXECUTION_PROFILES_PATH, "--execution-profiles", help="Path to execution profiles TOML"
    ),
    client_profiles: Path = typer.Option(
        CLIENT_PROFILES_PATH, "--client-profiles", help="Path to client profiles TOML"
    ),
):
    """
    Execute the entire research assessment and diagnostic pipeline end-to-end.
    """
    paper_stem, master_prompt, master_prompt_key = _resolve_inputs(
        paper_path, prompt_path, prompt_key, ground_truth_path
    )

    orchestrator = _setup_orchestrator(
        profile_name,
        execution_profiles,
        client_profile,
        client_profiles,
        paper_stem,
        master_prompt_key,
    )

    async def _execute():
        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_path
            )
            await _run_preprocess_stage(orchestrator, master_prompt, paper_context)
            await _run_assessment_stage(orchestrator, paper_context)
            if (
                orchestrator.profile.diagnostic
                and orchestrator.profile.diagnostic.prompt_source.value == "refined"
            ):
                refinement_result = orchestrator.step_executor.require_refinement_result()
                assessment_prompt = refinement_result.refined_prompt
            else:
                assessment_prompt = master_prompt
            await _run_diagnostic_stage(
                orchestrator, assessment_prompt, paper_context, ground_truth_path
            )
            await _run_results_stage(orchestrator, profile_name, master_prompt, ground_truth_path)
            logger.success("Pipeline complete.")
        finally:
            await orchestrator.provider.cleanup_context(paper_context)

    asyncio.run(_execute())


@app.command()
def run_stage(
    stage: str = typer.Argument(
        ..., help="Stage to run (preprocess, assessment, diagnostic, results)"
    ),
    client_profile: str = typer.Option(
        "unpaid", "--client-profile", help="Name of the client profile to use"
    ),
    profile_name: str = typer.Option("standard", "--profile", help="Name of the profile to use"),
    paper_path: Path = typer.Option(..., "--paper-path", help="Path to the source PDF"),
    prompt_path: Path = typer.Option(..., "--prompt-path", help="Path to the criteria file"),
    prompt_key: str | None = typer.Option(
        None, "--prompt-key", help="Key if using a registry file"
    ),
    ground_truth_path: Path = typer.Option(
        ..., "--ground-truth-path", help="Path to ground truth CSV"
    ),
    execution_profiles: Path = typer.Option(
        EXECUTION_PROFILES_PATH, "--execution-profiles", help="Path to execution profiles TOML"
    ),
    client_profiles: Path = typer.Option(
        CLIENT_PROFILES_PATH, "--client-profiles", help="Path to client profiles TOML"
    ),
):
    """
    Execute a single stage of the research pipeline.
    """
    paper_stem, master_prompt, master_prompt_key = _resolve_inputs(
        paper_path, prompt_path, prompt_key, ground_truth_path
    )

    orchestrator = _setup_orchestrator(
        profile_name,
        execution_profiles,
        client_profile,
        client_profiles,
        paper_stem,
        master_prompt_key,
    )

    async def _execute():
        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_path
            )

            if stage == "preprocess":
                await _run_preprocess_stage(orchestrator, master_prompt, paper_context)
            elif stage == "assessment":
                await _run_assessment_stage(orchestrator, paper_context)
            elif stage == "diagnostic":
                if (
                    orchestrator.profile.diagnostic
                    and orchestrator.profile.diagnostic.prompt_source.value == "refined"
                ):
                    refinement_result = orchestrator.step_executor.require_refinement_result()
                    assessment_prompt = refinement_result.refined_prompt
                else:
                    assessment_prompt = master_prompt
                await _run_diagnostic_stage(
                    orchestrator, assessment_prompt, paper_context, ground_truth_path
                )
            elif stage == "results":
                await _run_results_stage(
                    orchestrator, profile_name, master_prompt, ground_truth_path
                )
            else:
                logger.error(f"Stage '{stage}' is not supported.")
                raise typer.Exit(1)

        finally:
            await orchestrator.provider.cleanup_context(paper_context)

    asyncio.run(_execute())


@app.command()
def run_step(
    stage: str = typer.Argument(..., help="Stage (preprocess, assessment, diagnostic)"),
    step: str = typer.Argument(
        ..., help="Step to run (refine, extract, decompose, synthesize, analyze)"
    ),
    profile_name: str = typer.Option("standard", "--profile", help="Name of the profile to use"),
    client_profile: str = typer.Option(
        "unpaid", "--client-profile", help="Name of the client profile to use"
    ),
    paper_path: Path = typer.Option(..., "--paper-path", help="Path to the source PDF"),
    prompt_path: Path = typer.Option(..., "--prompt-path", help="Path to the criteria file"),
    prompt_key: str | None = typer.Option(
        None, "--prompt-key", help="Key if using a registry file"
    ),
    ground_truth_path: Path = typer.Option(
        ..., "--ground-truth-path", help="Path to ground truth CSV"
    ),
    execution_profiles: Path = typer.Option(
        EXECUTION_PROFILES_PATH, "--execution-profiles", help="Path to execution profiles TOML"
    ),
    client_profiles: Path = typer.Option(
        CLIENT_PROFILES_PATH, "--client-profiles", help="Path to client profiles TOML"
    ),
):
    """
    Execute a granular atomic step with strict prerequisite validation.
    """
    paper_stem, master_prompt, master_prompt_key = _resolve_inputs(
        paper_path, prompt_path, prompt_key, ground_truth_path
    )

    orchestrator = _setup_orchestrator(
        profile_name,
        execution_profiles,
        client_profile,
        client_profiles,
        paper_stem,
        master_prompt_key,
    )

    async def _execute():
        paper_context = None
        try:
            paper_context = await orchestrator.paper_context_service.build_initial_context(
                paper_path
            )

            if stage == "preprocess":
                await _run_preprocess_step(orchestrator, step, master_prompt, paper_context)

            elif stage == "assessment":
                await _run_assessment_step(orchestrator, step, paper_context)

            elif stage == "diagnostic":
                if (
                    orchestrator.profile.diagnostic
                    and orchestrator.profile.diagnostic.prompt_source.value == "refined"
                ):
                    refinement_result = orchestrator.step_executor.require_refinement_result()
                    assessment_prompt = refinement_result.refined_prompt
                else:
                    assessment_prompt = master_prompt
                await _run_diagnostic_step(
                    orchestrator, step, assessment_prompt, paper_context, ground_truth_path
                )

            else:
                logger.error(f"Stage '{stage}' is not supported.")
                raise typer.Exit(1)

        finally:
            await orchestrator.provider.cleanup_context(paper_context)

    asyncio.run(_execute())


async def _run_preprocess_step(
    orchestrator: MasterOrchestrator, step: str, master_prompt: str, paper_context
):
    """
    Internal dispatcher for preprocess-specific steps.

    Args:
        orchestrator: The active orchestrator instance.
        step: The name of the step to execute.
        master_prompt: The raw criteria text.
        paper_context: The document context.
    """
    if step == "refine":
        await orchestrator.execute_preprocess_refine(master_prompt)
        logger.success("Preprocess refinement complete.")

    elif step == "extract":
        await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
        await orchestrator.execute_preprocess_extraction(paper_context)
        logger.success("Preprocess extraction complete.")

    else:
        logger.error(f"Step '{step}' is not valid for the preprocess stage.")
        raise typer.Exit(1)


async def _run_preprocess_stage(
    orchestrator: MasterOrchestrator, master_prompt: str, paper_context
):
    """
    Execute the full preprocess stage.
    """
    from .core.enums import IngestionMode

    await _run_preprocess_step(orchestrator, "refine", master_prompt, paper_context)
    if paper_context.ingestion_mode == IngestionMode.EXTRACTION:
        await _run_preprocess_step(orchestrator, "extract", master_prompt, paper_context)


async def _run_assessment_step(orchestrator: MasterOrchestrator, step: str, paper_context):
    """
    Internal dispatcher for assessment-specific steps.
    """
    if step == "fast":
        refinement_result = orchestrator.step_executor.require_refinement_result()
        await orchestrator.paper_context_service.prepare_for_model_execution(paper_context)
        await orchestrator.paper_context_service.ensure_application_programming_interface_cache(
            paper_context, orchestrator.profile.assessment.synthesis.model.value
        )
        await orchestrator.execute_fast_assessment(paper_context, refinement_result.refined_prompt)
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
        evidence_reports = orchestrator.step_executor.require_assessment_evidence_reports(task_list)
        for group, evidence in zip(task_list.groups, evidence_reports):
            await orchestrator.execute_assessment_synthesis(group, evidence)
        logger.success("Assessment synthesis complete for all groups.")

    else:
        logger.error(f"Step '{step}' is not valid for the assessment stage.")
        raise typer.Exit(1)


async def _run_assessment_stage(orchestrator: MasterOrchestrator, paper_context):
    """
    Execute the full assessment stage.
    """
    from .core.enums import FragmentationMode

    if orchestrator.profile.assessment.fragmentation == FragmentationMode.FAST:
        await _run_assessment_step(orchestrator, "fast", paper_context)
    else:
        await _run_assessment_step(orchestrator, "decompose", paper_context)
        await _run_assessment_step(orchestrator, "extract", paper_context)
        await _run_assessment_step(orchestrator, "synthesize", paper_context)


async def _run_diagnostic_step(
    orchestrator: MasterOrchestrator,
    step: str,
    assessment_prompt: str,
    paper_context,
    ground_truth_path: Path | None,
):
    """
    Internal dispatcher for diagnostic-specific steps.
    """
    if not orchestrator.profile.diagnostic:
        logger.error("No diagnostic profile defined in the active recipe.")
        raise typer.Exit(1)

    assessment_report = await orchestrator.reconstruct_assessment_report()
    if not assessment_report:
        logger.error("Missing Assessment artifacts. Run 'assessment' stage first.")
        raise typer.Exit(1)

    ground_truth_data = _load_ground_truth_from_path(ground_truth_path, orchestrator.paper_stem)
    identifier_mapping = orchestrator.step_executor.get_identifier_mapping()
    assessment_details = [answer.model_dump() for answer in assessment_report.answers]

    strategy_target = orchestrator.profile.diagnostic.analysis.strategy
    filtered_assessment_details = orchestrator.step_executor.filter_diagnostic_details(
        assessment_details, strategy_target, identifier_mapping, ground_truth_data
    )

    if not filtered_assessment_details:
        logger.info("No assessment_details matched the diagnostic target criteria. Cannot proceed.")
        raise typer.Exit(1)

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
        logger.error(f"Step '{step}' is not valid for the diagnostic stage.")
        raise typer.Exit(1)


async def _run_diagnostic_stage(
    orchestrator: MasterOrchestrator,
    assessment_prompt: str,
    paper_context,
    ground_truth_path: Path | None,
):
    """
    Execute the full diagnostic stage.
    """
    from .core.enums import FragmentationMode

    if not orchestrator.profile.diagnostic:
        return
    if orchestrator.profile.diagnostic.fragmentation == FragmentationMode.FAST:
        await _run_diagnostic_step(
            orchestrator, "fast", assessment_prompt, paper_context, ground_truth_path
        )
    else:
        await _run_diagnostic_step(
            orchestrator, "decompose", assessment_prompt, paper_context, ground_truth_path
        )
        await _run_diagnostic_step(
            orchestrator, "analyze", assessment_prompt, paper_context, ground_truth_path
        )


async def _run_results_stage(
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
    ground_truth_data = _load_ground_truth_from_path(
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

    base_filename = (
        f"{profile_name}_{orchestrator.paper_stem}_{orchestrator.master_prompt_key}_{settings_hex}"
    )

    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, "w") as result_file:
        result_file.write(final_result.model_dump_json(indent=2))

    md_path = output_dir / f"{base_filename}.md"
    with open(md_path, "w") as report_file:
        report_file.write(markdown_report)

    logger.success(f"Generated results saved to {output_dir}/")


@database_app.command("clear")
def database_clear():
    """
    Wipe the active cache and run history from the local database.
    """
    store = ArtifactStore()
    store.clear_database()
    typer.echo("Database wiped successfully.")


@database_app.command("clear-stage")
def database_clear_stage(
    stage: PipelineStage = typer.Argument(..., help="Stage to clear")
):
    """
    Remove all artifacts belonging to a specific stage from the active cache.
    """
    store = ArtifactStore()
    store.clear_stage(stage.value)
    typer.echo(f"Stage '{stage.value}' cleared successfully.")


@database_app.command("seed")
def database_seed():
    """
    Re-inject pre-calculated artifacts from resources/convenience into the database.
    """
    from .convenience import restore_default_convenience_data

    restore_default_convenience_data()
    typer.echo("Convenience artifacts restored.")


@database_app.command("capture")
def database_capture(
    keys: list[str] | None = typer.Argument(
        None, help="Specific artifact keys to capture. If omitted, captures all entries."
    ),
):
    """
    Export database artifacts into the resources/convenience directory.
    """
    from .convenience import capture_current_artifacts

    capture_current_artifacts(keys=keys)
    typer.echo("Artifacts captured to resources/convenience.")


@database_app.command("clear-convenience")
def database_clear_convenience():
    """
    Wipe all serialized artifacts from the resources/convenience directory.
    """
    from .convenience import clear_convenience_data

    clear_convenience_data()
    typer.echo("Convenience directory wiped.")


@database_app.command("status")
def database_status():
    """
    Show current database statistics and entry counts.
    """
    store = ArtifactStore()
    import sqlite3

    try:
        with sqlite3.connect(store.database_path) as connection:
            active_count = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        typer.echo(f"Active Cached Artifacts: {active_count}")
        typer.echo(f"Run History Entries: {run_count}")
    except sqlite3.OperationalError:
        typer.echo("Database not yet initialized or does not exist.")


if __name__ == "__main__":
    app()
