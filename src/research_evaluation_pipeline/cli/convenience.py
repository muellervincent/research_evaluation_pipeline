"""
Convenience utilities for manual data injection and maintenance.
"""

import json
from pathlib import Path

from loguru import logger

from ..core.artifact_store import ArtifactStore

CONVENIENCE_ARTIFACTS_DIR = Path("resources/convenience")


def restore_default_convenience_data():
    """
    Restores default manual artifacts from resources/convenience into the database.

    This utility is used during database maintenance to re-inject pre-calculated
    results for common papers/prompts, allowing developers to skip redundant
    initial pipeline stages.
    """

    if not CONVENIENCE_ARTIFACTS_DIR.exists():
        logger.warning(
            f"Manual data directory not found: {CONVENIENCE_ARTIFACTS_DIR}. Skipping restoration."
        )
        return

    store = ArtifactStore()

    for file_path in CONVENIENCE_ARTIFACTS_DIR.iterdir():
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        key = file_path.stem

        try:
            if file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    content = json.load(f)
                store.save_artifact(key, content)
                logger.info(f"Restored JSON artifact: {key}")

        except Exception as error:
            logger.error(f"Failed to restore artifact from {file_path.name}: {error}")


def capture_current_artifacts(keys: list[str] | None = None):
    """
    Captures specific or all currently cached artifacts from the database and saves them to resources/convenience.

    This allows developers to 'freeze' a successful pipeline run (or specific stages) and use it as
    a baseline for future development without re-executing long-running stages.

    Args:
        keys: Optional list of specific artifact keys to capture. If None, captures all.
    """
    store = ArtifactStore()

    if keys:
        artifacts = []
        for key in keys:
            content = store.get_artifact(key)
            if content:
                artifacts.append((key, content))
            else:
                logger.warning(f"Artifact not found for key: {key}")
    else:
        artifacts = store.get_all_artifacts()

    if not artifacts:
        logger.info("No artifacts found to capture.")
        return

    CONVENIENCE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for key, content in artifacts:
        file_path = CONVENIENCE_ARTIFACTS_DIR / f"{key}.json"
        with open(file_path, "w") as f:
            json.dump(content, f, indent=2)
        logger.info(f"Captured artifact: {key}")


def clear_convenience_data():
    """
    Wipes all files from the resources/convenience directory while preserving the folder.
    """
    if not CONVENIENCE_ARTIFACTS_DIR.exists():
        logger.info("Convenience directory does not exist. Nothing to clear.")
        return

    count = 0
    for file_path in CONVENIENCE_ARTIFACTS_DIR.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            file_path.unlink()
            count += 1

    logger.warning(f"Cleared {count} convenience artifacts from {CONVENIENCE_ARTIFACTS_DIR}.")
