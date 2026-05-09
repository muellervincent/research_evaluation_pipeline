"""
Unit tests for the GeminiProvider implementation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from research_evaluation_pipeline.clients.gemini_provider import GeminiProvider


class MockResponse(BaseModel):
    """Simple model for testing structured output."""

    answer: str


@pytest.mark.asyncio
async def test_generate_text_output():
    """
    Verify that generate_text_output correctly calls the GenAI client.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello from Gemini"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GeminiProvider(client=mock_client)

    result = await provider.generate_text_output(
        model_name="gemini-2.0-flash", prompt_text="Say hello"
    )

    assert result == "Hello from Gemini"
    mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_structured_output():
    """
    Verify that generate_structured_output correctly parses JSON from the model.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"answer": "Yes"}'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GeminiProvider(client=mock_client)

    result = await provider.generate_structured_output(
        model_name="gemini-2.0-flash", prompt_text="Is this a test?", response_model=MockResponse
    )

    assert isinstance(result, MockResponse)
    assert result.answer == "Yes"
    mock_client.aio.models.generate_content.assert_called_once()
