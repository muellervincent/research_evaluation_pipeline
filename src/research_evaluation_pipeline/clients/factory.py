"""
Factory module for instantiating model providers.
"""

from typing import Any

import keyring
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..core.enums import GeminiModelName, OpenAIModelName
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .provider_protocol import ModelProvider

_GEMINI_SERVICE_NAME = "gemini_api_key_prompt_optimization"
_GEMINI_ACCOUNT_NAME = "odonata.vmueller"

_OPENAI_SERVICE_NAME = "openai_api_key_prompt_optimization"
_OPENAI_ACCOUNT_NAME = "mueller.vincent"


class MultiClientProvider(ModelProvider):
    """
    A routing provider that automatically delegates requests to either the
    GeminiProvider or OpenAIProvider based on the requested model name.
    """

    def __init__(self):
        """
        Initialize the underlying providers with hardcoded keychain parameters.
        """
        self._gemini = self._init_gemini()
        self._openai = self._init_openai()

    def _init_gemini(self) -> GeminiProvider:
        api_key = keyring.get_password(_GEMINI_SERVICE_NAME, _GEMINI_ACCOUNT_NAME)
        if not api_key:
            raise ValueError("Could not find Gemini API key in keychain.")
        return GeminiProvider(client=genai.Client(api_key=api_key))

    def _init_openai(self) -> OpenAIProvider:
        api_key = keyring.get_password(_OPENAI_SERVICE_NAME, _OPENAI_ACCOUNT_NAME)
        if not api_key:
            raise ValueError("Could not find OpenAI API key in keychain.")
        return OpenAIProvider(client=AsyncOpenAI(api_key=api_key))

    def _get_provider(self, model_name: str | GeminiModelName | OpenAIModelName) -> ModelProvider:
        """
        Retrieve the appropriate provider based on the model name.
        """
        if isinstance(model_name, GeminiModelName) or model_name in [
            m.value for m in GeminiModelName
        ]:
            return self._gemini
        elif isinstance(model_name, OpenAIModelName) or model_name in [
            m.value for m in OpenAIModelName
        ]:
            return self._openai
        raise ValueError(f"Unknown model name: {model_name}")

    @property
    def provider_type(self) -> str:
        return "multi"

    async def generate_structured_output(
        self,
        model_name: str | GeminiModelName | OpenAIModelName,
        prompt_text: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: Any | None = None,
    ) -> BaseModel:
        provider = self._get_provider(model_name)
        model_name_str = model_name.value if hasattr(model_name, "value") else str(model_name)
        return await provider.generate_structured_output(
            model_name=model_name_str,
            prompt_text=prompt_text,
            response_model=response_model,
            temperature=temperature,
            system_instruction=system_instruction,
            file_references=file_references,
            paper_context=paper_context,
        )

    async def generate_text_output(
        self,
        model_name: str | GeminiModelName | OpenAIModelName,
        prompt_text: str,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: Any | None = None,
    ) -> str:
        provider = self._get_provider(model_name)
        model_name_str = model_name.value if hasattr(model_name, "value") else str(model_name)
        return await provider.generate_text_output(
            model_name=model_name_str,
            prompt_text=prompt_text,
            temperature=temperature,
            system_instruction=system_instruction,
            file_references=file_references,
            paper_context=paper_context,
        )

    async def delete_cache(self, cache_name: str) -> None:
        """
        Delegates cache deletion. Cache names are usually provider-specific,
        but since we don't know the provider, we attempt on both.
        """
        try:
            await self._gemini.delete_cache(cache_name)
        except Exception:
            pass
        try:
            await self._openai.delete_cache(cache_name)
        except Exception:
            pass

    async def cache_content(
        self, model_name: str | GeminiModelName | OpenAIModelName, content: Any
    ) -> str | None:
        provider = self._get_provider(model_name)
        model_name_str = model_name.value if hasattr(model_name, "value") else str(model_name)
        return await provider.cache_content(model_name_str, content)

    async def cleanup_context(self, context: Any) -> None:
        """
        Cleans up the context across all initialized providers.
        """
        await self._gemini.cleanup_context(context)
        await self._openai.cleanup_context(context)

    async def upload_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Uploading a file generically is not supported. Use upload_file_to_provider instead.
        """
        raise NotImplementedError("Use upload_file_to_provider instead.")

    async def upload_file_to_provider(
        self, model_name: str | GeminiModelName | OpenAIModelName, file_bytes: bytes, filename: str
    ) -> str:
        provider = self._get_provider(model_name)
        return await provider.upload_file(file_bytes, filename)

    async def validate_file(self, file_id: str) -> bool:
        """
        Validates the file across providers (not usually called directly on MultiClientProvider).
        """
        raise NotImplementedError("Use validate_file_on_provider instead.")

    async def validate_file_on_provider(
        self, model_name: str | GeminiModelName | OpenAIModelName, file_id: str
    ) -> bool:
        provider = self._get_provider(model_name)
        return await provider.validate_file(file_id)

    def get_provider_type(self, model_name: str | GeminiModelName | OpenAIModelName) -> str:
        provider = self._get_provider(model_name)
        return provider.provider_type
