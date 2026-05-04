"""
Diagnostic-specific data structures.
"""

from pydantic import BaseModel, Field


class DiagnosticTask(BaseModel):
    """
    Representation of a single prediction targeted for diagnostic analysis.
    """

    question_id: str
    criteria: str
    model_answer: bool
    model_justification: str
    ground_truth_answer: bool | None


class DiagnosticGroup(BaseModel):
    """
    A logical cluster of diagnostic tasks grouped for cohesive investigation.
    """

    group_name: str
    tasks: list[DiagnosticTask]


class DiagnosticTaskList(BaseModel):
    """
    The output of the diagnostic decomposition stage, defining the plan for a fragmented run.
    """

    groups: list[DiagnosticGroup]


class DiagnosticItem(BaseModel):
    """
    A diagnostic classification and detailed explanation for a single prediction error.
    """

    question_id: str
    category: str = Field(
        description="A short, descriptive category (max 5 words) explaining the error, e.g., 'Model Hallucination', 'Ground Truth Error', 'Ambiguous Source Text'."
    )
    explanation: str = Field(
        description="Detailed diagnostic explanation comparing the model's logic, the ground truth, and the objective source text."
    )


class DiagnosticReport(BaseModel):
    """
    The final consolidated output of the diagnostic stage, containing all classifications.
    """

    analyses: list[DiagnosticItem]
