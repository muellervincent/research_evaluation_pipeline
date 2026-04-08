from typing import List, Optional
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
    answer: bool = Field(..., description="Boolean indicating if the practice was observed (Yes/No)")
    justification: str = Field(..., description="The textual quote or short explanatory text acting as justification.")

class AssessmentReportArtifact(BaseModel):
    answers: List[AssessmentAnswer] = Field(..., description="The definitive answers to the task list based on the extracted evidence.")
