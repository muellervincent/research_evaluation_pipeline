"""
Typer CLI for the Research Pipeline.
"""

import asyncio
from pathlib import Path

import typer
from loguru import logger

from .core.artifact_store import ArtifactStore
from .core.enums import PipelineStage
from .runner import PipelineRunner, RunnerError

EXECUTION_PROFILES_PATH = Path("resources/profiles/execution.toml")
CLIENT_PROFILES_PATH = Path("resources/profiles/client.toml")

app = typer.Typer(help="Research Reporting Pipeline CLI", no_args_is_help=True)
database_app = typer.Typer(help="Database management commands", no_args_is_help=True)
app.add_typer(database_app, name="db")


@app.command()
def run_pipeline(
    profile_name: str = typer.Option(
        "standard", "--profile", help="Name of the strategy profile to use"
    ),
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
    runner = PipelineRunner(
        execution_profiles_path=execution_profiles, client_profiles_path=client_profiles
    )

    try:
        asyncio.run(
            runner.run_pipeline(
                paper_path=paper_path,
                prompt_path=prompt_path,
                ground_truth_path=ground_truth_path,
                profile_name=profile_name,
                client_profile_name=client_profile,
                prompt_key=prompt_key,
            )
        )
    except RunnerError as error:
        logger.error(error)
        raise typer.Exit(1)


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
    runner = PipelineRunner(
        execution_profiles_path=execution_profiles, client_profiles_path=client_profiles
    )

    try:
        asyncio.run(
            runner.run_stage(
                stage=stage,
                paper_path=paper_path,
                prompt_path=prompt_path,
                ground_truth_path=ground_truth_path,
                profile_name=profile_name,
                client_profile_name=client_profile,
                prompt_key=prompt_key,
            )
        )
    except RunnerError as error:
        logger.error(error)
        raise typer.Exit(1)


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
    runner = PipelineRunner(
        execution_profiles_path=execution_profiles, client_profiles_path=client_profiles
    )

    try:
        asyncio.run(
            runner.run_step(
                stage=stage,
                step=step,
                paper_path=paper_path,
                prompt_path=prompt_path,
                ground_truth_path=ground_truth_path,
                profile_name=profile_name,
                client_profile_name=client_profile,
                prompt_key=prompt_key,
            )
        )
    except RunnerError as error:
        logger.error(error)
        raise typer.Exit(1)


@database_app.command("clear")
def database_clear():
    """
    Wipe the active cache and run history from the local database.
    """
    store = ArtifactStore()
    store.clear_database()
    typer.echo("Database wiped successfully.")


@database_app.command("clear-stage")
def database_clear_stage(stage: PipelineStage = typer.Argument(..., help="Stage to clear")):
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
