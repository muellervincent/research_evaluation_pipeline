"""
Client configuration settings for the research evaluation pipeline.
Defines how to retrieve credentials and instantiate specific LLM clients.
"""

from pydantic import BaseModel, Field

from ..core.enums import ClientType


class ClientProfile(BaseModel):
    """
    Configuration for an LLM client provider.

    Specifies the provider type and the necessary metadata to retrieve
    API keys from the system keychain.
    """

    client_type: ClientType = Field(..., description="The type of LLM provider to instantiate")
    description: str | None = Field(None, description="A human-readable label for this profile")
    keychain_service: str = Field(..., description="The service name used in the macOS keychain")
    keychain_account: str = Field(..., description="The account name used in the macOS keychain")
