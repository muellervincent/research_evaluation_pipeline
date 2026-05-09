"""
Protocol defining the interface for an LLM provider.
"""

from typing import Any, Protocol

from pydantic import BaseModel


class ModelProvider(Protocol):
    """
    Protocol defining the interface for an LLM provider.
    """

    @property
    def provider_type(self) -> str:
        """Return the name/type of the provider (e.g., 'gemini', 'openai')."""
        ...

    async def generate_structured_output(
        self,
        model_name: str,
        prompt_text: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: Any | None = None,
    ) -> BaseModel:
        """Generate structured output from the model based on a Pydantic schema."""
        ...

    async def generate_text_output(
        self,
        model_name: str,
        prompt_text: str,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: Any | None = None,
    ) -> str:
        """Generate raw text output from the model."""
        ...

    async def delete_cache(self, cache_name: str) -> None:
        """Delete a context cache from the provider."""
        ...

    async def cache_content(self, model_name: str, content: Any) -> str | None:
        """Generic method to cache large artifacts (PaperContext, text blocks, etc.) on the provider API."""
        ...

    async def cleanup_context(self, context: Any) -> None:
        """Safely tears down all remote resources (caches, files) associated with a context object."""
        ...

    async def upload_file(self, file_bytes: bytes, filename: str) -> str:
        """Upload a file to the provider's storage and return a unique identifier."""
        ...

    async def validate_file(self, file_id: str) -> bool:
        """Check if a file still exists and is usable on the provider's server."""
        ...
