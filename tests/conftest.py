"""
Shared fixtures for the research pipeline test suite.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_evaluation_pipeline.clients.provider_protocol import ModelProvider
from research_evaluation_pipeline.config.execution_settings import (
    AssessmentDecompositionSettings,
    AssessmentExtractionSettings,
    AssessmentProfile,
    AssessmentSynthesisSettings,
    DiagnosticAnalysisSettings,
    DiagnosticDecompositionSettings,
    DiagnosticProfile,
    PipelineProfile,
    PreprocessProfile,
    RefinementSettings,
    StepSettings,
)
from research_evaluation_pipeline.config.prompt_registry import PromptRegistry, PromptTemplate
from research_evaluation_pipeline.core.artifact_store import ArtifactStore
from research_evaluation_pipeline.core.enums import (
    DiagnosticPromptSource,
    FragmentationMode,
    IngestionMode,
    ModelName,
)
from research_evaluation_pipeline.service.artifact_key_builder import ArtifactKeyBuilder
from research_evaluation_pipeline.service.master_orchestrator import MasterOrchestrator
from research_evaluation_pipeline.service.paper_context_service import PaperContextService
from research_evaluation_pipeline.service.step_executor import StepExecutor


@pytest.fixture
def mock_provider():
    """Provides a mocked ModelProvider for LLM interactions."""
    provider = MagicMock(spec=ModelProvider)
    provider.generate_structured_output = AsyncMock()
    provider.generate_text_output = AsyncMock()
    provider.upload_file = AsyncMock()
    provider.delete_file = AsyncMock()
    provider.delete_cache = AsyncMock()
    provider.cache_content = AsyncMock()
    provider.cleanup_context = AsyncMock()
    return provider


@pytest.fixture
def in_memory_store():
    """Provides an ArtifactStore backed by a temporary file."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    db_path = Path(path)
    store = ArtifactStore(database_path=db_path)
    yield store
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_prompt_registry():
    """Provides a mocked PromptRegistry."""
    registry = MagicMock(spec=PromptRegistry)
    registry.get_prompt.return_value = PromptTemplate(
        system_text="System instruction",
        user_text="User prompt {variable}",
        coordinates="test.prompt",
    )
    return registry


@pytest.fixture
def pipeline_profile():
    """Provides a realistic PipelineProfile for testing."""
    return PipelineProfile(
        ingestion_mode=IngestionMode.MD,
        preprocess=PreprocessProfile(
            refinement=RefinementSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
            extraction=StepSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
        ),
        assessment=AssessmentProfile(
            fragmentation=FragmentationMode.PLAN,
            decomposition=AssessmentDecompositionSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
            extraction=AssessmentExtractionSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
            synthesis=AssessmentSynthesisSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
        ),
        diagnostic=DiagnosticProfile(
            fragmentation=FragmentationMode.PLAN,
            prompt_source=DiagnosticPromptSource.REFINED,
            decomposition=DiagnosticDecompositionSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
            analysis=DiagnosticAnalysisSettings(model=ModelName.GEMINI_3_FLASH_PREVIEW),
        ),
    )


@pytest.fixture
def orchestrator(mock_provider, pipeline_profile, mock_prompt_registry, in_memory_store):
    """Provides a MasterOrchestrator instance with mocked dependencies."""
    key_builder = ArtifactKeyBuilder(
        profile=pipeline_profile, paper_stem="test_paper", master_prompt_key="test_prompt"
    )
    paper_context_service = PaperContextService(
        provider=mock_provider,
        artifact_store=in_memory_store,
        key_builder=key_builder,
        profile=pipeline_profile,
    )
    step_executor = StepExecutor(
        artifact_store=in_memory_store, key_builder=key_builder, profile=pipeline_profile
    )

    return MasterOrchestrator(
        provider=mock_provider,
        profile=pipeline_profile,
        prompt_registry=mock_prompt_registry,
        artifact_store=in_memory_store,
        key_builder=key_builder,
        paper_context_service=paper_context_service,
        step_executor=step_executor,
        paper_stem="test_paper",
        master_prompt_key="test_prompt",
    )
