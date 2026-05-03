"""
Domain entities for the research assessment and diagnostic pipeline.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from .enums import IngestionMode


class PaperContext(BaseModel):
    """
    Encapsulates the state and data of a research paper throughout the pipeline.

    This provides a unified context to the provider models, decoupling them from raw file paths.
    It manages the lifecycle of the paper from raw PDF to uploaded API objects and model-specific caches.
    """

    pdf_path: Path
    ingestion_mode: IngestionMode

    raw_text: str | None = None

    raw_bytes: bytes | None = Field(None, exclude=True)

    uploaded_file: Any | None = Field(None, exclude=True)

    model_caches: dict[str, str] = Field(default_factory=dict)

    @computed_field
    @property
    def paper_stem(self) -> str:
        """
        Extract the base filename (stem) of the source PDF.

        Returns:
            The filename without extension.
        """
        return Path(self.pdf_path).stem

    def has_model_cache(self, model_name: str) -> bool:
        """
        Check if a context cache exists for a specific model.

        Args:
            model_name: The identifier of the AI model.

        Returns:
            True if a cache reference is stored for this model.
        """
        return model_name in self.model_caches

    def get_model_cache(self, model_name: str) -> str | None:
        """
        Retrieve the cache identifier for a specific model.

        Args:
            model_name: The identifier of the AI model.

        Returns:
            The cache name string if it exists, otherwise None.
        """
        return self.model_caches.get(model_name)
