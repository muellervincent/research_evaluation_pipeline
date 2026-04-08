import json
from pathlib import Path
from pydantic_settings import BaseSettings

import keyring

# SERVICE_NAME = "gemini_api_key_prompt_optimization"
# ACCOUNT_NAME = "vincent.emanuel.mueller"

# SERVICE_NAME = "gemini_api_key_default_gemini_project"
# ACCOUNT_NAME = "odonata.vmueller"

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
    # model: str = "gemini-3.1-pro-preview"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.2

    gemini_api_key: str = get_api_key(service_name, account_name)

    class Config:
        env_file = ".env"

settings = Settings()

def load_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
         data = json.load(f)
         return data.get("prompt", "")
