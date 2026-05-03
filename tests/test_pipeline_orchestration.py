"""
Integration tests for the MasterOrchestrator and pipeline logic.
"""

import pytest
from pydantic import BaseModel

from research_evaluation_pipeline.core.enums import CachePolicy
from research_evaluation_pipeline.logic.assessment.schemas import (
    AssessmentAnswer,
    AssessmentEvidenceReport,
    AssessmentGroup,
    AssessmentReport,
    AssessmentTask,
    AssessmentTaskList,
)
from research_evaluation_pipeline.logic.preprocess.schemas import RefinementResult


class MockResult(BaseModel):
    """Simple model for testing pipeline results."""

    data: str


@pytest.mark.asyncio
async def test_orchestrator_cache_miss_and_hit(orchestrator, mock_provider, in_memory_store):
    """
    Verify that the orchestrator correctly uses the cache.
    """
    key = "test_step_key"
    policy = CachePolicy.USE_CACHE
    expected_data = "Fresh Data"

    mock_provider.generate_structured_output.return_value = MockResult(data=expected_data)

    async def dummy_step():
        return MockResult(data=expected_data)

    result1 = await orchestrator._execute_with_cache(
        key=key, policy=policy, response_model=MockResult, coroutine=dummy_step()
    )

    assert result1.data == expected_data
    assert in_memory_store.get_artifact(key) == {"data": expected_data}

    mock_provider.generate_structured_output.return_value = MockResult(data="Should not be seen")

    async def dummy_step_fail():
        pytest.fail("Coroutine should not have been executed on cache hit")

    result2 = await orchestrator._execute_with_cache(
        key=key, policy=policy, response_model=MockResult, coroutine=dummy_step_fail()
    )

    assert result2.data == expected_data
    assert result2 == result1


@pytest.mark.asyncio
async def test_orchestrator_bypass_cache(orchestrator, mock_provider, in_memory_store):
    """
    Verify that the orchestrator bypasses the cache when policy is BYPASS.
    """
    key = "bypass_key"
    in_memory_store.save_artifact(key, {"data": "Old Data"})

    new_data = "New Data"

    async def new_step():
        return MockResult(data=new_data)

    result = await orchestrator._execute_with_cache(
        key=key, policy=CachePolicy.BYPASS_CACHE, response_model=MockResult, coroutine=new_step()
    )

    assert result.data == new_data
    cached = in_memory_store.get_artifact(key)
    assert cached == {"data": "Old Data"}

@pytest.mark.asyncio
async def test_reconstruct_assessment_report(orchestrator, in_memory_store):
    """
    Verify that reconstruct_assessment_report correctly merges fragments from cache.
    """
    prompt = "Refined Prompt"
    refinement = RefinementResult(refined_prompt=prompt)
    in_memory_store.save_artifact(orchestrator.key_builder.preprocess_refine_key(), refinement.model_dump())

    task_list = AssessmentTaskList(
        groups=[
            AssessmentGroup(group_name="Group1", tasks=[AssessmentTask(question_id="Q1", question_text="T1")]),
            AssessmentGroup(group_name="Group2", tasks=[AssessmentTask(question_id="Q2", question_text="T2")]),
        ]
    )
    in_memory_store.save_artifact(orchestrator.key_builder.assessment_decompose_key(prompt), task_list.model_dump())

    evidence1 = AssessmentEvidenceReport(group_name="Group1", evidence_items=[])
    in_memory_store.save_artifact(orchestrator.key_builder.assessment_extract_key(task_list.groups[0]), evidence1.model_dump())
    
    report1 = AssessmentReport(answers=[AssessmentAnswer(question_id="Q1", answer=True, justification="J1")])
    in_memory_store.save_artifact(orchestrator.key_builder.assessment_synthesize_key("Group1", evidence1), report1.model_dump())

    evidence2 = AssessmentEvidenceReport(group_name="Group2", evidence_items=[])
    in_memory_store.save_artifact(orchestrator.key_builder.assessment_extract_key(task_list.groups[1]), evidence2.model_dump())
    
    report2 = AssessmentReport(answers=[AssessmentAnswer(question_id="Q2", answer=False, justification="J2")])
    in_memory_store.save_artifact(orchestrator.key_builder.assessment_synthesize_key("Group2", evidence2), report2.model_dump())

    final_report = await orchestrator.reconstruct_assessment_report()

    assert final_report is not None
    assert len(final_report.answers) == 2
    assert final_report.answers[0].question_id == "Q1"
    assert final_report.answers[0].answer is True
    assert final_report.answers[1].question_id == "Q2"
    assert final_report.answers[1].answer is False
