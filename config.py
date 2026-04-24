import json

import keyring
from pydantic_settings import BaseSettings

# Not linked to billing account
# SERVICE_NAME = "gemini_api_key_default_gemini_project"
# ACCOUNT_NAME = "odonata.vmueller"

# Linked to billing account
# SERVICE_NAME = "gemini_api_key_prompt_optimization"
# ACCOUNT_NAME = "odonata.vmueller"


def get_api_key(service_name, account_name):
    key = keyring.get_password(service_name, account_name)

    if key:
        return key
    else:
        raise ValueError("API key not found in keychain")


class Settings(BaseSettings):
    service_name: str = "gemini_api_key_prompt_optimization"
    account_name: str = "odonata.vmueller"
    MODEL_CODE: str = "gemini-3.1-pro-preview"
    reasoning_model: str = MODEL_CODE
    extraction_model: str = MODEL_CODE
    refinement_model: str = MODEL_CODE
    temperature: float = 0.0

    gemini_api_key: str = get_api_key(service_name, account_name)

    class Config:
        env_file = ".env"


settings = Settings()


def load_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("prompt", "")
