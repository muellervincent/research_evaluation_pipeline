"""
Resource loaders for the research pipeline.
Handles all filesystem interactions (reading PDF bytes, loading TOML/YAML/CSV)
to keep the downstream runner stateless and path-agnostic.
"""

import csv
from pathlib import Path

import tomli
import yaml

from ..config.client_settings import ClientProfile
from ..config.execution_settings import PipelineProfile


class ResourceLoaderError(Exception):
    """Exception raised when resource loading fails."""


def load_execution_profile(profiles_path: Path, profile_name: str) -> PipelineProfile:
    """
    Load and validate a PipelineProfile from a TOML file.
    """
    if not profiles_path.exists():
        raise ResourceLoaderError(f"Profiles file not found: {profiles_path}")

    with open(profiles_path, "rb") as profile_file:
        profiles_data = tomli.load(profile_file)

    if profile_name not in profiles_data:
        raise ResourceLoaderError(f"Profile '{profile_name}' not found in {profiles_path}")

    try:
        return PipelineProfile(**profiles_data[profile_name])
    except Exception as error:
        raise ResourceLoaderError(f"Failed to validate PipelineProfile '{profile_name}': {error}")


def load_client_profile(profiles_path: Path, profile_name: str) -> ClientProfile:
    """
    Load and validate a ClientProfile from a TOML file.
    """
    if not profiles_path.exists():
        raise ResourceLoaderError(f"Client profiles file not found: {profiles_path}")

    with open(profiles_path, "rb") as profile_file:
        profiles_data = tomli.load(profile_file)

    if profile_name not in profiles_data:
        raise ResourceLoaderError(f"Client profile '{profile_name}' not found in {profiles_path}")

    try:
        return ClientProfile(**profiles_data[profile_name])
    except Exception as error:
        raise ResourceLoaderError(f"Failed to validate ClientProfile '{profile_name}': {error}")


def load_paper(paper_path: Path) -> tuple[str, bytes]:
    """
    Load the paper bytes and stem from a PDF file.
    """
    if not paper_path.exists():
        raise ResourceLoaderError(f"Paper file not found: {paper_path}")
    return paper_path.stem, paper_path.read_bytes()


def load_prompt(prompt_path: Path, prompt_key: str | None = None) -> str:
    """
    Load the master prompt text from MD, TXT, or YAML.
    """
    if not prompt_path.exists():
        raise ResourceLoaderError(f"Prompt file not found: {prompt_path}")

    extension = prompt_path.suffix.lower()
    with open(prompt_path, "r") as prompt_file:
        if extension in [".yaml", ".yml"]:
            data = yaml.safe_load(prompt_file)
            if prompt_key:
                if isinstance(data, dict) and prompt_key in data:
                    return str(data[prompt_key])
                raise ResourceLoaderError(f"Prompt key '{prompt_key}' not found in {prompt_path}")
            if isinstance(data, str):
                return data
            if isinstance(data, dict) and len(data) == 1:
                return str(list(data.values())[0])
            raise ResourceLoaderError("Prompt key is required for registry files.")

        if extension in [".txt", ".md"]:
            return prompt_file.read().strip()

        raise ResourceLoaderError(f"Unsupported prompt file extension: {extension}")


def load_ground_truth(ground_truth_path: Path | None, paper_stem: str) -> dict[str, bool] | None:
    """
    Load ground truth answers for a specific paper from a CSV.
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
    except Exception as error:
        # We don't want to crash the whole pipeline if ground truth fails to load
        # as it is only needed for diagnostic analysis targeting mismatches.
        from loguru import logger

        logger.error(f"Failed to load ground truth from {ground_truth_path}: {error}")
        return None
