"""
Prompt template management for the research evaluation and diagnostic pipeline.
"""

import yaml
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "resources" / "prompts_default.yaml"


class PromptTemplate:
    """
    Represents an AI prompt template capable of variable interpolation.

    Encapsulates both the system instruction and the user message, allowing for
    dynamic formatting of both segments.
    """

    def __init__(self, system_text: str, user_text: str, coordinates: str):
        """
        Initialize a prompt template with its raw text and source location.

        Args:
            system_text: The system instruction template.
            user_text: The user message template.
            coordinates: The dot-separated path identifying the prompt in the YAML file.
        """
        self.system_text = system_text
        self.user_text = user_text
        self.coordinates = coordinates

    def format(self, **kwargs) -> "PromptTemplate":
        """
        Inject variables into the system and user templates.

        Args:
            **kwargs: Key-value pairs for string interpolation.

        Returns:
            A new PromptTemplate instance containing the formatted text.

        Raises:
            ValueError: If a required placeholder is missing from the provided variables.
        """
        try:
            formatted_system = self.system_text.format(**kwargs) if self.system_text else ""
            formatted_user = self.user_text.format(**kwargs) if self.user_text else ""
            return PromptTemplate(formatted_system, formatted_user, self.coordinates)
        except KeyError as error:
            raise ValueError(
                f"Missing required variable {error} for prompt template at {self.coordinates}"
            )


class PromptService:
    """
    Loader and access layer for the consolidated prompt repository.

    Parses the 'prompts_default.yaml' file and provides a structured interface
    for retrieving templates by their logical path.
    """

    def __init__(self, yaml_path: Path = DEFAULT_PROMPTS_PATH):
        """
        Initialize the service by loading all templates from the filesystem.

        Args:
            yaml_path: The path to the YAML file containing all prompts.
        """
        self.yaml_path = yaml_path
        self._prompts: dict[str, Any] = self._load_all_prompts()

    def _load_all_prompts(self) -> dict[str, Any]:
        """
        Read the entire prompt tree from the source YAML file.

        Returns:
            A nested dictionary representing the prompt hierarchy.
        """
        if not self.yaml_path.exists():
            logger.error(f"Prompts YAML file not found at: {self.yaml_path}")
            return {}

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as error:
            logger.error(f"Failed to load prompts YAML: {error}")
            return {}

    def get_prompt(self, path: str) -> PromptTemplate:
        """
        Retrieve a specific prompt template by its logical dot-separated path.

        Args:
            path: The coordinates of the prompt (e.g., 'assessment.decomposition.semantic').

        Returns:
            A PromptTemplate instance populated with the raw template strings.

        Raises:
            KeyError: If the specified path does not exist in the YAML file.
            ValueError: If the path points to a non-dictionary leaf node.
        """
        keys = path.split(".")
        data = self._prompts
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                raise KeyError(f"Prompt path not found: {path} (missing key: {key})")
            data = data[key]

        if not isinstance(data, dict):
            raise ValueError(f"Prompt path does not point to a template dictionary: {path}")

        system = data.get("system") or ""
        user = data.get("user") or ""
        return PromptTemplate(system, user, path)
