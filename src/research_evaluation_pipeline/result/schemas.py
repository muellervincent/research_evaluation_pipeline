"""
Data structures for the final pipeline results.
"""

from pydantic import BaseModel


class QuestionResult(BaseModel):
    """
    Consolidated result for a single question, merging assessment and diagnostic details.
    """

    semantic_id: str
    original_id: str
    ground_truth_answer: bool | None
    model_answer: bool | None
    is_correct: bool | None
    justification: str | None
    diagnostic_category: str | None = None
    diagnostic_explanation: str | None = None


class PipelineMetrics(BaseModel):
    """
    Summary metrics for the pipeline execution.
    """

    total_questions: int
    correct_predictions: int
    accuracy: float


class FinalPipelineResult(BaseModel):
    """
    The final consolidated output of the research pipeline.
    """

    paper_stem: str
    master_prompt_key: str
    metrics: PipelineMetrics
    executed_at: str
    refined_prompt: str | None = None
    results: list[QuestionResult]
