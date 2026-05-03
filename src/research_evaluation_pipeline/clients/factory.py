"""
Factory module for instantiating model providers.
"""

import keyring
from google import genai

from ..config.client_settings import ClientProfile
from ..core.enums import ClientType
from .gemini import GeminiProvider
from .provider_protocol import ModelProvider


class ProviderFactory:
    """
    Factory for creating and configuring model providers.

    It abstracts the instantiation of specific LLM providers and handles secure
    credential retrieval from the keychain.
    """

    @staticmethod
    def get_provider(profile: ClientProfile) -> ModelProvider:
        """
        Create a provider instance using the provided client profile.

        Retrieves the necessary API keys from the macOS keychain using the
        service and account identifiers specified in the profile.

        Args:
            profile: The ClientProfile containing provider type and keychain metadata.

        Returns:
            An implementation of the ModelProvider protocol.

        Raises:
            ValueError: If the API key cannot be retrieved from the keychain or the
                client type is unsupported.
        """

        try:
            api_key = keyring.get_password(profile.keychain_service, profile.keychain_account)
        except Exception:
            api_key = None

        if not api_key:
            raise ValueError(
                f"Could not find API key for {profile.client_type} in keychain ({profile.keychain_service})."
            )

        if profile.client_type == ClientType.GEMINI:
            client = genai.Client(api_key=api_key)
            return GeminiProvider(client=client)

        raise ValueError(f"Unsupported client type: {profile.client_type}")
