"""
Gemini implementation of the ModelProvider protocol.
"""

from typing import Any

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel

from ..core.domain import PaperContext
from ..core.enums import IngestionMode
from .provider_protocol import ModelProvider


class GeminiProvider(ModelProvider):
    """
    Model provider implementation using the Google GenAI Software Development Kit.

    Handles high-level model interactions including structured generation,
    file uploads, and automated context caching management.
    """

    MAX_OUTPUT_TOKENS = 32768
    MIN_CACHE_TOKENS = 8192
    CACHE_TIME_TO_LIVE = "400s"

    def __init__(self, client: genai.Client):
        """
        Initialize the provider with a configured Google GenAI client.

        Args:
            client: The GenAI client instance.
        """
        self.client = client

    async def generate_structured_output(
        self,
        model_name: str,
        prompt_text: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: PaperContext | None = None,
    ) -> BaseModel:
        """
        Request a structured JSON response from the model.

        Args:
            model_name: The identifier of the Gemini model.
            prompt_text: The user instruction.
            response_model: The Pydantic model for validation and schema generation.
            temperature: The sampling temperature.
            system_instruction: Optional developer-level guidance.
            file_references: Optional list of additional media files.
            paper_context: The primary document context.

        Returns:
            An instance of response_model populated with the model's judgment.
        """

        cache_name = paper_context.get_model_cache(model_name) if paper_context else None

        if paper_context and cache_name and system_instruction:
            logger.debug(
                "Merging system instruction into user prompt due to Context Caching constraints."
            )
            prompt_text = f"{system_instruction}\n\nUSER PROMPT:\n{prompt_text}"
            system_instruction = None

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_model,
            system_instruction=system_instruction,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )

        if paper_context and cache_name:
            config.cached_content = cache_name

        contents = self._build_contents(model_name, prompt_text, file_references, paper_context)

        response = await self.client.aio.models.generate_content(
            model=model_name, contents=contents, config=config
        )

        logger.debug(
            f"Received response for {model_name}. Length: {len(response.text)} characters."
        )
        if hasattr(response, "usage_metadata"):
            logger.debug(
                f"Token usage: {response.usage_metadata.candidates_token_count} output tokens."
            )

        return response_model.model_validate_json(response.text)

    async def generate_text_output(
        self,
        model_name: str,
        prompt_text: str,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: PaperContext | None = None,
    ) -> str:
        """
        Request a raw text response from the model.

        Args:
            model_name: The identifier of the Gemini model.
            prompt_text: The user instruction.
            temperature: The sampling temperature.
            system_instruction: Optional developer-level guidance.
            file_references: Optional additional media files.
            paper_context: The primary document context.

        Returns:
            The raw text response from the model.
        """

        cache_name = paper_context.get_model_cache(model_name) if paper_context else None

        if paper_context and cache_name and system_instruction:
            logger.debug(
                "Merging system instruction into user prompt due to Context Caching constraints."
            )
            prompt_text = f"{system_instruction}\n\nUSER PROMPT:\n{prompt_text}"
            system_instruction = None

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )

        if paper_context and cache_name:
            config.cached_content = cache_name

        contents = self._build_contents(model_name, prompt_text, file_references, paper_context)

        response = await self.client.aio.models.generate_content(
            model=model_name, contents=contents, config=config
        )

        logger.debug(
            f"Received text response for {model_name}. Length: {len(response.text)} characters."
        )
        if hasattr(response, "usage_metadata"):
            logger.debug(
                f"Token usage: {response.usage_metadata.candidates_token_count} output tokens."
            )

        return response.text

    def _build_contents(
        self,
        model_name: str,
        prompt_text: str,
        file_references: list[Any] | None = None,
        paper_context: PaperContext | None = None,
    ) -> list[types.Part]:
        """
        Assemble the content parts for a generation request.

        Args:
            model_name: The model being targeted.
            prompt_text: The user prompt.
            file_references: Additional files.
            paper_context: The primary document.

        Returns:
            A list of types.Part objects ready for the API call.
        """

        contents = []

        if paper_context and not paper_context.has_model_cache(model_name):
            if paper_context.ingestion_mode == IngestionMode.EXTRACTION and paper_context.raw_text:
                contents.append(
                    types.Part.from_text(
                        text=f"RESEARCH PAPER MARKDOWN:\n\n{paper_context.raw_text}"
                    )
                )
            elif paper_context.uploaded_file:
                contents.append(paper_context.uploaded_file)
                contents.append(
                    types.Part.from_text(
                        text="I have provided the research paper as a file attachment for your analysis."
                    )
                )
            elif paper_context.raw_text:
                contents.append(
                    types.Part.from_text(
                        text=f"RESEARCH PAPER MARKDOWN:\n\n{paper_context.raw_text}"
                    )
                )
            elif paper_context.raw_bytes:
                contents.append(
                    types.Part.from_bytes(data=paper_context.raw_bytes, mime_type="application/pdf")
                )

        if file_references:
            for reference in file_references:
                if isinstance(reference, bytes):
                    contents.append(
                        types.Part.from_bytes(data=reference, mime_type="application/pdf")
                    )
                else:
                    contents.append(reference)

        contents.append(types.Part.from_text(text=prompt_text))
        return contents

    async def upload_file(self, file_path: str) -> Any:
        """
        Upload a file to the Gemini File Application Programming Interface.

        Args:
            file_path: Absolute path to the local file.

        Returns:
            The file metadata object returned by the API.
        """
        logger.info(f"Uploading file to Gemini File Application Programming Interface: {file_path}")
        file_result = await self.client.aio.files.upload(file=file_path)
        return file_result

    async def delete_file(self, file_reference: Any) -> None:
        """
        Remove a file from the Gemini File API.

        Args:
            file_reference: The file object or identifier.
        """
        if file_reference and hasattr(file_reference, "name"):
            logger.info(f"Deleting Gemini file reference: {file_reference.name}")
            try:
                await self.client.aio.files.delete(name=file_reference.name)
            except Exception as error:
                logger.error(f"Failed to delete file {file_reference.name}: {error}")

    async def delete_cache(self, cache_name: str) -> None:
        """
        Remove a Context Cache from the Application Programming Interface.

        Args:
            cache_name: The unique resource name of the cache.
        """
        if cache_name:
            logger.info(f"Deleting Gemini Context Cache: {cache_name}")
            try:
                await self.client.aio.caches.delete(name=cache_name)
            except Exception as error:
                logger.error(f"Failed to delete cache {cache_name}: {error}")

    async def _cache_content(self, model_name: str, contents: list[Any]) -> str | None:
        """
        Calculate tokens and create a cache resource if appropriate.

        Args:
            model_name: The model to associate the cache with.
            contents: The parts to be cached.

        Returns:
            The cache resource name if created, else None.
        """
        try:
            token_response = await self.client.aio.models.count_tokens(
                model=model_name, contents=contents
            )
            if token_response.total_tokens >= self.MIN_CACHE_TOKENS:
                logger.info(
                    f"Content is {token_response.total_tokens} tokens. Creating Context Cache..."
                )
                cache = await self.client.aio.caches.create(
                    model=model_name,
                    config=types.CreateCachedContentConfig(contents=contents, ttl=self.CACHE_TIME_TO_LIVE),
                )
                logger.info(f"Context Cache created successfully. Name: {cache.name}")
                return cache.name

            logger.info("Content below token minimum. Skipping cache.")
            return None
        except Exception as error:
            logger.warning(f"Caching failed. Falling back to raw content. Error: {error}")
            return None

    async def cache_content(self, model_name: str, content: Any) -> str | None:
        """
        Public method to cache various document types (PaperContext or strings).

        Args:
            model_name: The targeted model.
            content: The data to cache.

        Returns:
            The generated cache name if successful.
        """
        contents_to_cache = []

        if isinstance(content, PaperContext):
            if content.ingestion_mode == IngestionMode.EXTRACTION and content.raw_text:
                contents_to_cache.append(
                    types.Part.from_text(text=f"RESEARCH PAPER MARKDOWN:\n\n{content.raw_text}")
                )
            elif content.uploaded_file:
                contents_to_cache.append(content.uploaded_file)
                contents_to_cache.append(
                    types.Part.from_text(
                        text="I have provided the research paper as a file attachment for your analysis."
                    )
                )
            elif content.raw_text:
                contents_to_cache.append(
                    types.Part.from_text(text=f"RESEARCH PAPER MARKDOWN:\n\n{content.raw_text}")
                )
            elif content.raw_bytes:
                contents_to_cache.append(
                    types.Part.from_bytes(data=content.raw_bytes, mime_type="application/pdf")
                )

            cache_name = await self._cache_content(model_name, contents_to_cache)
            if cache_name:
                content.model_caches[model_name] = cache_name
            return cache_name

        elif isinstance(content, str):
            contents_to_cache.append(types.Part.from_text(text=content))
            return await self._cache_content(model_name, contents_to_cache)

        else:
            logger.warning(f"Unsupported content type for caching: {type(content)}")
            return None

    async def cleanup_context(self, context: Any) -> None:
        """
        Tear down all remote resources (caches, files) associated with a paper.

        Args:
            context: The document context to clean up.
        """
        if not isinstance(context, PaperContext):
            return

        logger.info(f"Running pipeline resource cleanup for {context.paper_stem}...")
        if context.model_caches:
            for cache_model, cache_name in list(context.model_caches.items()):
                await self.delete_cache(cache_name)
                del context.model_caches[cache_model]

        if context.uploaded_file:
            await self.delete_file(context.uploaded_file)
            context.uploaded_file = None
        logger.info("Pipeline resource cleanup complete.")
