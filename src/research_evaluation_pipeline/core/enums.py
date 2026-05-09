"""
Global enums for the research evaluation and diagnostic pipeline.
"""

from enum import Enum


class SanitizedEnum(str, Enum):
    """
    Base enum providing string sanitization for use in CLI flags and database keys.

    Inherits from str to ensure JSON serializability.
    """

    @property
    def value_sanitized(self) -> str:
        """
        Return the enum value with characters normalized for filesystem and URL safety.

        Replaces dots and underscores with dashes.

        Returns:
            The sanitized string value.
        """
        return self.value.replace(".", "-").replace("_", "-")


class DiagnosticPromptSource(SanitizedEnum):
    """
    Source of the prompt used for diagnostic analysis.
    """

    MASTER = "master"
    REFINED = "refined"


class PipelineStage(SanitizedEnum):
    """
    Major execution phases of the research pipeline.
    """

    PREPROCESS = "preprocess"
    ASSESSMENT = "assessment"
    DIAGNOSTIC = "diagnostic"


class FragmentationMode(SanitizedEnum):
    """
    Granularity levels for data processing and reasoning.
    """

    FAST = "fast"
    PLAN = "plan"


class IngestionMode(SanitizedEnum):
    """
    Methods for providing research paper content to the AI model.
    """

    MD = "md"
    PDF = "pdf"


class ProcessingMode(SanitizedEnum):
    """
    Concurrency models for executing multi-task extraction steps.
    """

    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"


class CachePolicy(SanitizedEnum):
    """
    Rules for interacting with the persistent ArtifactStore.
    """

    USE_CACHE = "use-cache"
    BYPASS_CACHE = "bypass-cache"
    OVERWRITE_CACHE = "overwrite-cache"


class RefinementStrategy(SanitizedEnum):
    """
    Algorithms for cleaning and structuring the master criteria.
    """

    STANDARD = "standard"
    SEMANTIC = "semantic"


class AssessmentDecompositionStrategy(SanitizedEnum):
    """
    Strategies for breaking down assessment criteria into logical task groups.
    """

    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


class DiagnosticDecompositionStrategy(SanitizedEnum):
    """
    Strategies for grouping predictions for diagnostic analysis.
    """

    THEMATIC = "thematic"


class AssessmentExtractionStrategy(SanitizedEnum):
    """
    Strategies for locating and extracting verbatim evidence for assessment.
    """

    STANDARD = "standard"


class DiagnosticAnalysisStrategy(SanitizedEnum):
    """
    Strategies for diagnostic analysis of predictions.
    """

    DIAGNOSE_ALL = "diagnose-all"
    DIAGNOSE_MISMATCHES = "diagnose-mismatches"
    DIAGNOSE_MATCHES = "diagnose-matches"
    DIAGNOSE_OVERPREDICTIONS = "diagnose-overpredictions"
    DIAGNOSE_UNDERPREDICTIONS = "diagnose-underpredictions"


class AssessmentSynthesisStrategy(SanitizedEnum):
    """
    Reasoning frameworks for rendering final assessment decisions.
    """

    CONCISE = "concise"
    ANALYTICAL = "analytical"
    VERBOSE = "verbose"


class GeminiModelName(SanitizedEnum):
    """
    Supported Google Gemini model identifiers.
    """

    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_PRO = "gemini-2.0-pro"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_1_FLASH_LITE_PREVIEW = "gemini-3.1-flash-lite-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"


class OpenAIModelName(SanitizedEnum):
    """
    Supported OpenAI model identifiers.
    """

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"
    O3_MINI = "o3-mini"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest"


class ClientType(SanitizedEnum):
    """
    Supported LLM client providers.
    """

    GEMINI = "gemini"
    OPENAI = "openai"
