"""
Deterministic generation of final JSON and Markdown reports.
"""

import hashlib
import json
from datetime import datetime

from ..config.execution_settings import PipelineProfile
from ..logic.assessment.schemas import AssessmentReport
from ..logic.diagnostic.schemas import DiagnosticReport
from .schemas import FinalPipelineResult, PipelineMetrics, QuestionResult


class ResultBuilder:
    """
    Generates structured JSON and human-readable Markdown reports from pipeline artifacts.
    """

    @staticmethod
    def build_final_result(
        profile: PipelineProfile,
        assessment_report: AssessmentReport,
        identifier_mapping: dict[str, str | None],
        ground_truth: dict[str, bool] | None,
        paper_stem: str,
        master_prompt_key: str,
        diagnostic_report: DiagnosticReport | None = None,
        refined_prompt: str | None = None,
    ) -> FinalPipelineResult:
        """
        Merge all execution data into a single structured result object.
        """
        results = []
        correct_prediction_count = 0
        total_question_count = 0

        diagnostic_lookup_table = {}
        if diagnostic_report:
            for analysis_item in diagnostic_report.analyses:
                diagnostic_lookup_table[analysis_item.question_id] = analysis_item

        for assessment_answer in assessment_report.answers:
            original_identifier = (
                identifier_mapping.get(assessment_answer.question_id)
                or assessment_answer.question_id
            )
            expected_answer = ground_truth.get(original_identifier) if ground_truth else None

            is_correct_answer = None
            if expected_answer is not None:
                is_correct_answer = assessment_answer.answer == expected_answer
                total_question_count += 1
                if is_correct_answer:
                    correct_prediction_count += 1

            diagnostic_item = diagnostic_lookup_table.get(assessment_answer.question_id)
            diagnostic_category = diagnostic_item.category if diagnostic_item else None
            diagnostic_explanation = diagnostic_item.explanation if diagnostic_item else None

            results.append(
                QuestionResult(
                    semantic_id=assessment_answer.question_id,
                    original_id=original_identifier,
                    ground_truth_answer=expected_answer,
                    model_answer=assessment_answer.answer,
                    is_correct=is_correct_answer,
                    justification=assessment_answer.justification,
                    diagnostic_category=diagnostic_category,
                    diagnostic_explanation=diagnostic_explanation,
                )
            )

        accuracy = (
            (correct_prediction_count / total_question_count) if total_question_count > 0 else 0.0
        )

        metrics = PipelineMetrics(
            total_questions=total_question_count,
            correct_predictions=correct_prediction_count,
            accuracy=accuracy,
        )

        return FinalPipelineResult(
            paper_stem=paper_stem,
            master_prompt_key=master_prompt_key,
            metrics=metrics,
            executed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            refined_prompt=refined_prompt,
            results=results,
        )

    @staticmethod
    def build_markdown_report(result: FinalPipelineResult, profile: PipelineProfile) -> str:
        """
        Generate a human-readable Markdown string from the final result.
        """
        report_lines = []
        report_lines.append(f"# Pipeline Execution Report: {result.paper_stem}")
        report_lines.append("")
        report_lines.append(f"**Result Created:** {result.executed_at}")
        report_lines.append("")

        report_lines.append("## Execution Settings")
        report_lines.append("```json")
        report_lines.append(json.dumps(profile.model_dump(mode="json"), indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.append("## Metrics")
        report_lines.append(f"- **Total Scored Questions:** {result.metrics.total_questions}")
        report_lines.append(f"- **Correct Predictions:** {result.metrics.correct_predictions}")
        report_lines.append(f"- **Accuracy:** {result.metrics.accuracy * 100:.2f}%")
        report_lines.append("")

        report_lines.append("## Detailed Results")

        report_lines.append(
            "| ID | Original ID | Ground Truth | Model Answer | Correct | Diagnostic Category |"
        )
        report_lines.append("|---|---|---|---|---|---|")
        for question_result in result.results:
            ground_truth_display = (
                str(question_result.ground_truth_answer)
                if question_result.ground_truth_answer is not None
                else "N/A"
            )
            model_answer_display = (
                str(question_result.model_answer)
                if question_result.model_answer is not None
                else "N/A"
            )
            is_correct_display = (
                "✅"
                if question_result.is_correct is True
                else ("❌" if question_result.is_correct is False else "➖")
            )
            diagnostic_category_display = question_result.diagnostic_category or "-"
            report_lines.append(
                f"| {question_result.semantic_id} | {question_result.original_id} | {ground_truth_display} | {model_answer_display} | {is_correct_display} | {diagnostic_category_display} |"
            )

        report_lines.append("")
        report_lines.append("## Justifications & Explanations")
        for question_result in result.results:
            report_lines.append(
                f"### {question_result.semantic_id} ({question_result.original_id})"
            )
            report_lines.append(f"**Model Justification:**\n{question_result.justification}\n")
            if question_result.diagnostic_explanation:
                report_lines.append(
                    f"**Diagnostic Explanation ({question_result.diagnostic_category}):**\n{question_result.diagnostic_explanation}\n"
                )
            report_lines.append("---")

        if result.refined_prompt:
            report_lines.append("## Refined Prompt")
            report_lines.append("```markdown")
            report_lines.append(result.refined_prompt)
            report_lines.append("```")

        return "\n".join(report_lines)

    @staticmethod
    def get_settings_hex(profile: PipelineProfile) -> str:
        """
        Generate a hex hash from the profile settings.
        """
        configuration_content = profile.model_dump_json()
        return hashlib.sha256(configuration_content.encode("utf-8")).hexdigest()[:8]
