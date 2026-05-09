"""
Deterministic artifact key factory for the research assessment and diagnostic pipeline.

All artifact keys follow the pattern:
    {stage}-{step}__{flag_value}__{flag_value}...

Only semantically relevant flags are included per step, in fixed order:
model, stem, master_prompt_key, refinement_strategy, then step-specific discriminators.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from ..config.execution_settings import PipelineProfile


class ArtifactKeyBuilder:
    """
    Constructs deterministic, content-addressed artifact cache keys for every pipeline step.

    Key schema: ``{stage}-{step}__{flag_value}__...`` where flags are included only
    when semantically relevant to that specific step (e.g., stem is omitted for
    purely prompt-level steps that are independent of which paper is being processed).
    """

    def __init__(self, profile: PipelineProfile, paper_stem: str, master_prompt_key: str):
        """
        Initialize the builder with the active execution profile and instance identifiers.

        Args:
            profile: The configuration for the current run.
            paper_stem: The identifier for the paper being processed.
            master_prompt_key: The identifier for the assessment criteria.
        """
        self._profile = profile
        self._stem = paper_stem
        self._master_prompt_key = master_prompt_key
        self._refinement_strategy = profile.preprocess.refinement.strategy.value_sanitized
        self._ingestion_mode = profile.ingestion_mode.value_sanitized

    @property
    def paper_stem(self) -> str:
        """The identifier for the paper being processed."""
        return self._stem

    @property
    def master_prompt_key(self) -> str:
        """The identifier for the assessment criteria."""
        return self._master_prompt_key

    def _hash_input(self, data: Any) -> str:
        """
        Generate an 8-character hex digest of the input data.
        """
        if isinstance(data, BaseModel):
            content_bytes = data.model_dump_json().encode("utf-8")
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            content_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
        elif isinstance(data, bytes):
            content_bytes = data
        else:
            content_bytes = str(data).encode("utf-8")

        return hashlib.sha256(content_bytes).hexdigest()[:8]

    def preprocess_refine_key(self) -> str:
        """
        Generate the unique key for the prompt refinement artifact.

        Flags: model, master_prompt_key, refinement_strategy.
        Stem is excluded — refinement operates on the prompt, not the paper.

        Returns:
            The deterministic key string.
        """
        refinement_profile = self._profile.preprocess.refinement
        return f"preprocess-refine__{refinement_profile.model.value_sanitized}__{self._master_prompt_key}__{self._refinement_strategy}"

    def preprocess_extract_key(self) -> str:
        """
        Generate the unique key for the PDF-to-text extraction artifact.

        Flags: model, stem.
        Master prompt key and refinement strategy are excluded — extraction is a
        raw document conversion independent of the prompt pipeline.

        Returns:
            The deterministic key string.
        """
        extraction_profile = self._profile.preprocess.extraction
        return f"preprocess-extract__{extraction_profile.model.value_sanitized}__{self._stem}"

    def assessment_decompose_key(self, prompt: str) -> str:
        """
        Generate the unique key for the assessment task decomposition artifact.

        Flags: model, master_prompt_key, refinement_strategy, decomposition_strategy.
        Stem is excluded — decomposition is prompt-level, not paper-level.

        Returns:
            The deterministic key string.
        """
        decomposition_profile = self._profile.assessment.decomposition
        prompt_hash = self._hash_input(prompt)
        return (
            f"assessment-decompose"
            f"__{decomposition_profile.model.value_sanitized}"
            f"__{self._master_prompt_key}"
            f"__{self._refinement_strategy}"
            f"__{decomposition_profile.strategy.value_sanitized}"
            f"__{prompt_hash}"
        )

    def assessment_extract_key(self, group: Any) -> str:
        """
        Generate the unique key for assessment evidence extraction of a group.

        Flags: model, stem, master_prompt_key, refinement_strategy, extraction_strategy, group_name.
        Both paper-level (stem) and prompt-level flags are required here.

        Args:
            group: The logical task group (AssessmentGroup).

        Returns:
            The deterministic key string.
        """
        extraction_profile = self._profile.assessment.extraction
        group_hash = self._hash_input(group)
        return (
            f"assessment-extract"
            f"__{extraction_profile.model.value_sanitized}"
            f"__{self._stem}"
            f"__{self._master_prompt_key}"
            f"__{self._refinement_strategy}"
            f"__{self._ingestion_mode}"
            f"__{extraction_profile.strategy.value_sanitized}"
            f"__{group.group_name}"
            f"__{group_hash}"
        )

    def assessment_synthesize_key(self, group_name: str, evidence_report: Any) -> str:
        """
        Generate the unique key for assessment synthesis of a group.

        Flags: model, stem, master_prompt_key, refinement_strategy, synthesis_strategy, group_name.

        Args:
            group_name: The name of the logical task group.
            evidence_report: The AssessmentEvidenceReport object.

        Returns:
            The deterministic key string.
        """
        synthesis_profile = self._profile.assessment.synthesis
        evidence_hash = self._hash_input(evidence_report)
        return (
            f"assessment-synthesize"
            f"__{synthesis_profile.model.value_sanitized}"
            f"__{self._stem}"
            f"__{self._master_prompt_key}"
            f"__{self._refinement_strategy}"
            f"__{self._ingestion_mode}"
            f"__{synthesis_profile.strategy.value_sanitized}"
            f"__{group_name}"
            f"__{evidence_hash}"
        )

    def assessment_fast_key(self, prompt: str) -> str:
        """
        Generate the unique key for fast-mode (single-pass) assessment.

        Flags: model, stem, master_prompt_key, refinement_strategy, synthesis_strategy.

        Returns:
            The deterministic key string.
        """
        synthesis_profile = self._profile.assessment.synthesis
        prompt_hash = self._hash_input(prompt)
        return f"assessment-fast__{synthesis_profile.model.value_sanitized}__{self._stem}__{self._master_prompt_key}__{self._refinement_strategy}__{self._ingestion_mode}__{prompt_hash}"

    def diagnostic_decompose_key(self, prompt: str, assessment_details: Any) -> str:
        """
        Generate the unique key for diagnostic task decomposition.

        Flags: model, stem, master_prompt_key, refinement_strategy, decomposition_strategy, diagnostic_target.
        Stem is included (diagnostic is paper-specific). Target is a discriminator for what sub-population
        of predictions was analysed.

        Returns:
            The deterministic key string.
        """
        decomposition_profile = self._profile.diagnostic.decomposition
        input_hash = self._hash_input(
            f"{self._hash_input(prompt)}_{self._hash_input(assessment_details)}"
        )
        return (
            f"diagnostic-decompose"
            f"__{decomposition_profile.model.value_sanitized}"
            f"__{self._stem}"
            f"__{self._master_prompt_key}"
            f"__{self._refinement_strategy}"
            f"__{self._ingestion_mode}"
            f"__{decomposition_profile.strategy.value_sanitized}"
            f"__{self._profile.diagnostic.analysis.strategy.value_sanitized}"
            f"__{input_hash}"
        )

    def diagnostic_analyze_key(self, group: Any, prompt: str) -> str:
        """
        Generate the unique key for diagnostic analysis of a group.

        Flags: model, stem, master_prompt_key, refinement_strategy, analysis_strategy, group_name.

        Args:
            group: The logical task group (DiagnosticGroup).
            prompt: The criteria prompt.

        Returns:
            The deterministic key string.
        """
        analysis_profile = self._profile.diagnostic.analysis
        input_hash = self._hash_input(f"{self._hash_input(prompt)}_{self._hash_input(group)}")
        return (
            f"diagnostic-analyze"
            f"__{analysis_profile.model.value_sanitized}"
            f"__{self._stem}"
            f"__{self._master_prompt_key}"
            f"__{self._refinement_strategy}"
            f"__{self._ingestion_mode}"
            f"__{analysis_profile.strategy.value_sanitized}"
            f"__{group.group_name}"
            f"__{input_hash}"
        )

    def diagnostic_fast_key(self, prompt: str, assessment_details: Any) -> str:
        """
        Generate the unique key for fast-mode (single-pass) diagnostic.

        Flags: model, stem, master_prompt_key, refinement_strategy, synthesis_strategy.

        Returns:
            The deterministic key string.
        """
        analysis_profile = self._profile.diagnostic.analysis
        input_hash = self._hash_input(
            f"{self._hash_input(prompt)}_{self._hash_input(assessment_details)}"
        )
        return (
            f"diagnostic-fast"
            f"__{analysis_profile.model.value_sanitized}"
            f"__{self._stem}"
            f"__{self._master_prompt_key}"
            f"__{self._refinement_strategy}"
            f"__{self._ingestion_mode}"
            f"__{analysis_profile.strategy.value_sanitized}"
            f"__{input_hash}"
        )

    def paper_upload_key(self, raw_bytes: bytes | None = None) -> str:
        """
        Generate the unique key for the paper's uploaded file metadata.

        Flags: stem, and the hash of the raw bytes to ensure changes to the PDF trigger a re-upload.
        """
        if raw_bytes:
            return f"metadata-upload__{self._stem}__{self._hash_input(raw_bytes)}"
        return f"metadata-upload__{self._stem}"
