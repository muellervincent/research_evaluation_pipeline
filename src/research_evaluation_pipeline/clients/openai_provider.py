"""
OpenAI implementation of the ModelProvider protocol.
"""

import base64
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..core.paper_context import PaperContext
from .provider_protocol import ModelProvider


class OpenAIProvider(ModelProvider):
    """
    Model provider implementation using the OpenAI Python SDK.

    Supports structured output via JSON schemas and multimodal PDF input
    for vision-capable models like GPT-4o.
    """

    def __init__(self, client: AsyncOpenAI):
        """
        Initialize the provider with a configured OpenAI client.

        Args:
            client: The AsyncOpenAI client instance.
        """
        self.client = client

    @property
    def provider_type(self) -> str:
        """Return the name/type of the provider."""
        return "openai"

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
            model_name: The identifier of the OpenAI model (e.g., gpt-4o).
            prompt_text: The user instruction.
            response_model: The Pydantic model for validation and schema generation.
            temperature: The sampling temperature.
            system_instruction: Optional developer-level guidance.
            file_references: Optional list of additional files.
            paper_context: The primary document context.

        Returns:
            An instance of response_model populated with the model's judgment.
        """

        messages = self._build_messages(
            prompt_text, system_instruction, file_references, paper_context
        )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": self._map_pydantic_to_openai_schema(response_model),
            },
        }

        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty response.")

        logger.debug(
            f"Received structured response for {model_name}. Length: {len(content)} characters."
        )
        if hasattr(response, "usage"):
            logger.debug(f"Token usage: {response.usage.completion_tokens} output tokens.")

        return response_model.model_validate_json(content)

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
            model_name: The identifier of the OpenAI model.
            prompt_text: The user instruction.
            temperature: The sampling temperature.
            system_instruction: Optional developer-level guidance.
            file_references: Optional additional files.
            paper_context: The primary document context.

        Returns:
            The raw text response from the model.
        """

        messages = self._build_messages(
            prompt_text, system_instruction, file_references, paper_context
        )

        response = await self.client.chat.completions.create(
            model=model_name, messages=messages, temperature=temperature
        )

        content = response.choices[0].message.content
        if not content:
            return ""

        logger.debug(f"Received text response for {model_name}. Length: {len(content)} characters.")
        return content

    def _build_messages(
        self,
        prompt_text: str,
        system_instruction: str | None = None,
        file_references: list[Any] | None = None,
        paper_context: PaperContext | None = None,
    ) -> list[dict[str, Any]]:
        """
        Assemble the messages list for an OpenAI chat completion request.

        Args:
            prompt_text: The user prompt.
            system_instruction: The system prompt.
            file_references: Additional files.
            paper_context: The primary document.

        Returns:
            A list of message dictionaries.
        """

        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        user_content = []

        if paper_context:
            if paper_context.raw_text:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"RESEARCH PAPER MARKDOWN:\n\n{paper_context.raw_text}",
                    }
                )
            elif paper_context.raw_bytes:
                file_id = paper_context.uploaded_file_ids.get("openai")
                if file_id:
                    logger.debug(f"Using cached OpenAI file ID: {file_id}")
                    user_content.append({"type": "file", "file": {"file_id": file_id}})
                else:
                    logger.info(
                        f"Injecting PDF bytes for {paper_context.paper_stem} into OpenAI request."
                    )
                    b64_pdf = base64.b64encode(paper_context.raw_bytes).decode("utf-8")
                    user_content.append(
                        {
                            "type": "input_file",
                            "input_file": {
                                "filename": f"{paper_context.paper_stem}.pdf",
                                "file_data": f"data:application/pdf;base64,{b64_pdf}",
                            },
                        }
                    )

        if file_references:
            for reference in file_references:
                if isinstance(reference, bytes):
                    b64_ref = base64.b64encode(reference).decode("utf-8")
                    user_content.append(
                        {
                            "type": "input_file",
                            "input_file": {
                                "filename": "reference.pdf",
                                "file_data": f"data:application/pdf;base64,{b64_ref}",
                            },
                        }
                    )
                elif isinstance(reference, str):
                    user_content.append({"type": "text", "text": reference})

        user_content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _map_pydantic_to_openai_schema(self, model: type[BaseModel]) -> dict[str, Any]:
        """
        Convert a Pydantic model to an OpenAI-compatible JSON schema.

        OpenAI's 'strict' mode requires all fields to be required and
        'additionalProperties' set to false.

        Args:
            model: The Pydantic model class.

        Returns:
            A dictionary representing the JSON schema.
        """
        schema = model.model_json_schema()

        # OpenAI specific adjustments for strict mode
        self._clean_schema_for_openai(schema)

        return schema

    def _clean_schema_for_openai(self, schema: dict[str, Any]) -> None:
        """
        Recursively clean a JSON schema to meet OpenAI's strict requirements.

        - Sets additionalProperties to False for objects.
        - Ensures all properties are in 'required'.
        - Removes titles and other unsupported keys.

        Args:
            schema: The schema dictionary to modify in-place.
        """
        if not isinstance(schema, dict):
            return

        # Remove keys not supported or needed by OpenAI
        unsupported_keys = ["title", "description", "default", "examples"]
        for key in unsupported_keys:
            schema.pop(key, None)

        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            properties = schema.get("properties", {})
            if properties:
                schema["required"] = list(properties.keys())
                for prop_schema in properties.values():
                    self._clean_schema_for_openai(prop_schema)

        # Handle definitions
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                self._clean_schema_for_openai(def_schema)

        if "anyOf" in schema:
            for subschema in schema["anyOf"]:
                self._clean_schema_for_openai(subschema)

        if "allOf" in schema:
            for subschema in schema["allOf"]:
                self._clean_schema_for_openai(subschema)

        if schema.get("type") == "array" and "items" in schema:
            self._clean_schema_for_openai(schema["items"])

    async def delete_cache(self, cache_name: str) -> None:
        """
        No-op for OpenAI as caching is managed automatically.
        """
        pass

    async def cache_content(self, model_name: str, content: Any) -> str | None:
        """
        No-op for OpenAI as caching is managed automatically.
        """
        return "openai-automatic-cache"

    async def cleanup_context(self, context: Any) -> None:
        """
        No-op for OpenAI as there are no manual remote resources to clean up.
        """
        pass

    async def upload_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Upload a file to OpenAI's Files API.

        Args:
            file_bytes: The raw data to upload.
            filename: A label for the file.

        Returns:
            The unique file ID from OpenAI.
        """
        # OpenAI expects a file-like object or a tuple (filename, file_bytes)
        response = await self.client.files.create(file=(filename, file_bytes), purpose="vision")
        logger.info(f"Uploaded file to OpenAI: {response.id}")
        return response.id

    async def validate_file(self, file_id: str) -> bool:
        """
        Check if a file still exists in the OpenAI Files API.
        """
        try:
            await self.client.files.retrieve(file_id=file_id)
            return True
        except Exception:
            return False
