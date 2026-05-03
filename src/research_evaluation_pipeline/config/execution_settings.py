"""
Execution settings for the research reporting assessment pipeline.
Defines the granular hyperparameters and strategies for different pipeline steps.
"""

from pydantic import BaseModel, Field

from ..core.enums import (
    AssessmentDecompositionStrategy,
    AssessmentExtractionStrategy,
    AssessmentSynthesisStrategy,
    CachePolicy,
    DiagnosticAnalysisStrategy,
    DiagnosticDecompositionStrategy,
    DiagnosticPromptSource,
    FragmentationMode,
    IngestionMode,
    ModelName,
    ProcessingMode,
    RefinementStrategy,
)


class StepSettings(BaseModel):
    """
    Configuration for an atomic LLM model run.

    Defines the core hyperparameters and behavior for any single interaction with an AI model.
    """

    model: ModelName = Field(..., description="The identifier of the LLM model to use")
    temperature: float = Field(0.0, description="The sampling temperature for generation")
    cache_policy: CachePolicy = Field(
        CachePolicy.USE_CACHE, description="Rules for interacting with the ArtifactStore"
    )


class AssessmentDecompositionSettings(StepSettings):
    """
    Settings specifically for the assessment decomposition step.
    """

    strategy: AssessmentDecompositionStrategy = Field(
        AssessmentDecompositionStrategy.SEMANTIC,
        description="Decomposition strategy (Semantic/Structural)",
    )


class DiagnosticDecompositionSettings(StepSettings):
    """
    Settings specifically for the diagnostic decomposition step.
    """

    strategy: DiagnosticDecompositionStrategy = Field(
        DiagnosticDecompositionStrategy.THEMATIC, description="Decomposition strategy (Thematic)"
    )


class AssessmentExtractionSettings(StepSettings):
    """
    Settings for the assessment evidence extraction step.
    """

    strategy: AssessmentExtractionStrategy = Field(
        AssessmentExtractionStrategy.STANDARD, description="Evidence extraction strategy"
    )
    processing_mode: ProcessingMode = Field(
        ProcessingMode.CONCURRENT, description="Execution flow (Sequential or Concurrent)"
    )


class DiagnosticAnalysisSettings(StepSettings):
    """
    Settings for the diagnostic analysis step.
    """

    strategy: DiagnosticAnalysisStrategy = Field(
        DiagnosticAnalysisStrategy.DIAGNOSE_ALL, description="Diagnostic analysis target strategy"
    )
    processing_mode: ProcessingMode = Field(
        ProcessingMode.CONCURRENT, description="Execution flow (Sequential or Concurrent)"
    )


class AssessmentSynthesisSettings(StepSettings):
    """
    Settings for the final assessment synthesis step.
    """

    strategy: AssessmentSynthesisStrategy = Field(
        AssessmentSynthesisStrategy.ANALYTICAL, description="Synthesis reasoning strategy"
    )


class RefinementSettings(StepSettings):
    """
    Settings for the master prompt refinement step.
    """

    strategy: RefinementStrategy = Field(
        RefinementStrategy.STANDARD, description="Criteria cleaning strategy"
    )


class PreprocessProfile(BaseModel):
    """
    Full configuration for the preprocess stage.
    """

    refinement: RefinementSettings = Field(
        ..., description="Settings for cleaning the master criteria"
    )
    extraction: StepSettings = Field(..., description="Settings for converting the PDF to Markdown")


class AssessmentProfile(BaseModel):
    """
    Full configuration for the assessment stage.
    """

    fragmentation: FragmentationMode = Field(
        FragmentationMode.PLAN, description="The execution mode (FAST or PLAN)"
    )
    decomposition: AssessmentDecompositionSettings = Field(
        ..., description="Settings for criteria decomposition"
    )
    extraction: AssessmentExtractionSettings = Field(
        ..., description="Settings for evidence extraction"
    )
    synthesis: AssessmentSynthesisSettings = Field(..., description="Settings for final synthesis")


class DiagnosticProfile(BaseModel):
    """
    Full configuration for the diagnostic stage.
    """

    fragmentation: FragmentationMode = Field(
        FragmentationMode.PLAN, description="The execution mode (FAST or PLAN)"
    )
    prompt_source: DiagnosticPromptSource = Field(
        DiagnosticPromptSource.REFINED, description="Source of the criteria prompt"
    )
    decomposition: DiagnosticDecompositionSettings = Field(
        ..., description="Settings for diagnostic task decomposition"
    )
    analysis: DiagnosticAnalysisSettings = Field(
        ..., description="Settings for diagnostic analysis"
    )


class PipelineProfile(BaseModel):
    """
    The master configuration object for an entire pipeline execution.

    Encapsulates all stage-specific profiles and execution strategies.
    """

    ingestion_mode: IngestionMode = Field(
        ..., description="Method for providing paper content to models"
    )
    preprocess: PreprocessProfile = Field(..., description="Settings for the preprocess stage")
    assessment: AssessmentProfile = Field(..., description="Settings for the assessment stage")
    diagnostic: DiagnosticProfile | None = Field(
        None, description="Optional settings for the diagnostic stage"
    )
