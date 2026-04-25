from typing import List, Optional, Any
from pydantic import BaseModel, Field

class SubTask(BaseModel):
    question_number: str = Field(..., description="The ID or number of the question (e.g., '1', '7a')")
    question_text: str = Field(..., description="The full text of the question to evaluate")
    status: str = Field(default="Pending", description="Status of the subtask (Pending, In-Progress, Completed)")

class TaskGroup(BaseModel):
    group_name: str = Field(..., description="Name of the task group (e.g., 'Population and Sample Size')")
    subtasks: List[SubTask] = Field(..., description="List of specific questions in this group")

class TaskListArtifact(BaseModel):
    groups: List[TaskGroup] = Field(..., description="Logical groupings of evaluation questions")

class EvidenceItem(BaseModel):
    question_number: str = Field(..., description="The ID of the question this evidence targets")
    relevant_quotes: List[str] = Field(..., description="Exact quotes explicitly extracted from the text. If none exist, leave empty.")
    source_context: str = Field(..., description="Short explanation of where this was found in the text or why it's relevant.")

class EvidenceArtifact(BaseModel):
    group_name: str = Field(..., description="The task group name this evidence corresponds to")
    evidence_items: List[EvidenceItem] = Field(..., description="The extracted evidence per question")

class AssessmentAnswer(BaseModel):
    question_number: str = Field(..., description="The ID of the question")
    answer: bool = Field(..., description="Boolean indicating if the practice was observed (True/False)")
    justification: Optional[str] = Field(None, description="The textual quote or short explanatory text acting as justification. Required if running with justification enabled.")

class AssessmentReport(BaseModel):
    answers: List[AssessmentAnswer] = Field(..., description="The definitive answers for each question.")

class EvaluationDetail(BaseModel):
    question_number: str
    expected: str
    predicted: str
    correct: bool
    justification: Optional[str] = None
    mismatch_reason: Optional[str] = None
    mismatch_category: Optional[str] = None

class EvaluationMetrics(BaseModel):
    accuracy: float
    correct: int
    total: int
    details: List[EvaluationDetail]
    missing_answers: int = 0

class MismatchAnalysis(BaseModel):
    question_number: str = Field(..., description="The ID of the question that was incorrectly answered.")
    category: str = Field(..., description="The category of the mismatch (e.g., 'Model hallucinated/failed', 'Ambiguous Prompt', 'Better than Ground Truth', etc.)")
    explanation: str = Field(..., description="Detailed explanation of why the prediction and ground truth differ.")

class MismatchReport(BaseModel):
    analyses: List[MismatchAnalysis] = Field(..., description="List of mismatch analyses for incorrect answers.")


class EvaluationResult(BaseModel):
    mode: str
    pdf_stem: str
    model_name: str
    metrics: Optional[EvaluationMetrics] = None
    raw_output: Any = Field(None, description="The raw JSON Assessment string/dict.")
