"""
Data structures for the preprocess stage.
"""

from pydantic import BaseModel


class ExtractionResult(BaseModel):
    """
    The output of the extraction stage, containing the document text.
    """

    content: str
    paper_stem: str


class QuestionIdentifier(BaseModel):
    """
    A link between a model-generated semantic ID and its source document label.
    """

    semantic_id: str
    original_id: str | None = None


class RefinementResult(BaseModel):
    """
    The output of the refinement stage, containing structured criteria and ID mappings.
    """

    refined_prompt: str
    question_identifiers: list[QuestionIdentifier] | None = None

    def get_id_map(self) -> dict[str, str | None]:
        """
        Convert the list of question identifiers into a lookup dictionary.

        Returns:
            A dictionary mapping semantic IDs to their original labels.
        """
        if not self.question_identifiers:
            return {}
        return {qi.semantic_id: qi.original_id for qi in self.question_identifiers}
