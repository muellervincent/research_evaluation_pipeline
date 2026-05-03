"""
Assessment-specific data structures.
"""

from pydantic import BaseModel


class AssessmentTask(BaseModel):
    """
    Representation of a single inquiry within the assessment criteria.
    """
    question_id: str
    question_text: str


class AssessmentGroup(BaseModel):
    """
    A logical cluster of assessment tasks grouped for cohesive extraction and reasoning.
    """
    group_name: str
    tasks: list[AssessmentTask]


class AssessmentTaskList(BaseModel):
    """
    The output of the decomposition stage, defining the plan for a fragmented assessment run.
    """
    groups: list[AssessmentGroup]


class AssessmentEvidenceItem(BaseModel):
    """
    A collection of verbatim quotes and context for a specific assessment question.
    """
    question_id: str
    relevant_quotes: list[str]
    source_context: str


class AssessmentEvidenceReport(BaseModel):
    """
    The output of the extraction stage for a specific task group.
    """
    group_name: str
    evidence_items: list[AssessmentEvidenceItem]


class AssessmentAnswer(BaseModel):
    """
    A binary decision and reasoning for a single assessment question.
    """
    question_id: str
    answer: bool
    justification: str | None = None


class AssessmentReport(BaseModel):
    """
    The final consolidated output of the assessment stage, containing all judgments.
    """
    answers: list[AssessmentAnswer]
