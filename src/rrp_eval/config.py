import json
from pathlib import Path

import keyring
import tomli
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ProfileSettings(BaseSettings):
    profile_name: str = "default"
    description: str = ""
    keychain_service: str = ""
    keychain_account: str = ""
    reasoning_model: str = "gemini-2.5-flash"
    extraction_model: str = "gemini-2.5-flash"
    refinement_model: str = "gemini-2.5-flash"
    temperature: float = 0.8

    @property
    def gemini_api_key(self) -> str:
        if not self.keychain_service or not self.keychain_account:
            return ""
        key = keyring.get_password(self.keychain_service, self.keychain_account)
        if not key:
            raise ValueError(
                f"API key not found in keychain for {self.keychain_service}/{self.keychain_account}"
            )
        return key


class GlobalConfig:
    def __init__(self, toml_path: Path = PROJECT_ROOT / "eval_profiles.toml"):
        self.toml_path = toml_path
        self.profiles = {}
        self.load_profiles()
        self.active_profile_name = "default"
        self._settings = self._build_settings("default")

    def load_profiles(self):
        if self.toml_path.exists():
            with open(self.toml_path, "rb") as f:
                self.profiles = tomli.load(f)
        else:
            self.profiles = {}

    def _build_settings(self, profile_name: str) -> ProfileSettings:
        profile_data = self.profiles.get(profile_name, {})
        profile_data["profile_name"] = profile_name
        return ProfileSettings(**profile_data)

    def set_profile(self, profile_name: str):
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found in {self.toml_path}")
        self.active_profile_name = profile_name
        new_data = self._build_settings(profile_name).model_dump()
        for key, value in new_data.items():
            setattr(self._settings, key, value)

    @property
    def settings(self) -> ProfileSettings:
        return self._settings


# Singleton instance
config = GlobalConfig()
settings = config.settings


def load_prompt(prompt_file_path: str) -> str:
    path = Path(prompt_file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.suffix in [".md", ".txt"]:
        return path.read_text(encoding="utf-8")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("prompt", "")
