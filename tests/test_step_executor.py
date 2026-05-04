"""
Unit tests for the StepExecutor logic.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_evaluation_pipeline.core.enums import ProcessingMode
from research_evaluation_pipeline.core.paper_context import PaperContext
from research_evaluation_pipeline.logic.assessment.schemas import (
    AssessmentAnswer,
    AssessmentEvidenceReport,
    AssessmentGroup,
    AssessmentTask,
    AssessmentTaskList,
)


@pytest.mark.asyncio
async def test_dispatch_assessment_groups_sequential(orchestrator):
    """
    Verify that dispatch_assessment_groups correctly iterates through groups sequentially.
    """
    executor = orchestrator.step_executor
    orchestrator.profile.assessment.extraction.processing_mode = ProcessingMode.SEQUENTIAL

    task_list = AssessmentTaskList(
        groups=[
            AssessmentGroup(
                group_name="G1", tasks=[AssessmentTask(question_id="Q1", question_text="T1")]
            ),
            AssessmentGroup(
                group_name="G2", tasks=[AssessmentTask(question_id="Q2", question_text="T2")]
            ),
        ]
    )

    paper_context = PaperContext(paper_stem="test_paper")

    orchestrator.execute_assessment_extraction = AsyncMock(
        side_effect=[
            AssessmentEvidenceReport(group_name="G1", evidence_items=[]),
            AssessmentEvidenceReport(group_name="G2", evidence_items=[]),
        ]
    )
    orchestrator.execute_assessment_synthesis = AsyncMock(
        side_effect=[
            MagicMock(answers=[AssessmentAnswer(question_id="Q1", answer=True)]),
            MagicMock(answers=[AssessmentAnswer(question_id="Q2", answer=False)]),
        ]
    )

    answers = await executor.dispatch_assessment_groups(task_list, paper_context)

    assert len(answers) == 2
    assert answers[0].question_id == "Q1"
    assert answers[1].question_id == "Q2"
    assert orchestrator.execute_assessment_extraction.call_count == 2
    assert orchestrator.execute_assessment_synthesis.call_count == 2


@pytest.mark.asyncio
async def test_dispatch_assessment_groups_concurrent(orchestrator):
    """
    Verify that dispatch_assessment_groups correctly uses bounded concurrency.
    """
    executor = orchestrator.step_executor
    orchestrator.profile.assessment.extraction.processing_mode = ProcessingMode.CONCURRENT

    task_list = AssessmentTaskList(
        groups=[
            AssessmentGroup(group_name="G1", tasks=[]),
            AssessmentGroup(group_name="G2", tasks=[]),
            AssessmentGroup(group_name="G3", tasks=[]),
        ]
    )

    paper_context = PaperContext(paper_stem="test_paper")

    orchestrator.execute_assessment_extraction = AsyncMock(
        return_value=AssessmentEvidenceReport(group_name="any", evidence_items=[])
    )
    orchestrator.execute_assessment_synthesis = AsyncMock(
        return_value=MagicMock(answers=[AssessmentAnswer(question_id="any", answer=True)])
    )

    answers = await executor.dispatch_assessment_groups(task_list, paper_context)

    assert len(answers) == 3
    assert orchestrator.execute_assessment_extraction.call_count == 3
