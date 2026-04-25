import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, List, Union

from google import genai
from google.genai import types
from loguru import logger

from .config import settings
from .schema import (
    AssessmentReport,
    EvaluationDetail,
    EvidenceArtifact,
    MismatchReport,
    TaskListArtifact,
)


# Async helper to get client per call to respect active settings (since they can switch)
def get_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


async def process_pdf(pdf_path: str, cache_dir: Path = None) -> str:
    """Pre-processes a PDF, extracting exact text and describing images."""
    if cache_dir:
        stem = Path(pdf_path).stem
        cache_file = cache_dir / f"{stem}_{settings.extraction_model}.md"
        if cache_file.exists():
            logger.info(
                f"Using cached markdown for {stem} using {settings.extraction_model}"
            )
            return cache_file.read_text(encoding="utf-8")

    logger.info(
        f"Extracting text & image descriptions using {settings.extraction_model}"
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    extraction_prompt = """
    Extract all the text from this document exactly as it is, without summarization.
    Crucial rules for extraction:
    1. Maintain continuous text. Sentences must absolutely not be broken by page transitions.
    2. Completely remove all page structures, headers, footers, and page numbers.
    3. For any images, charts, or graphs present in the PDF, provide a plain text description of the visual information.
    4. If an image interrupts a sentence or paragraph, relocate the image description tag to the end of the surrounding paragraph to maintain text continuity.
    5. All original content must remain intact, just restructured for continuous reading.
    Format the overall output in Markdown.
    """

    client = get_client()
    response = await client.aio.models.generate_content(
        model=settings.extraction_model,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=extraction_prompt),
        ],
        config=types.GenerateContentConfig(temperature=0.1),
    )

    extracted_text = response.text
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(pdf_path).stem
        cache_file = cache_dir / f"{stem}_{settings.extraction_model}.md"
        cache_file.write_text(extracted_text, encoding="utf-8")

    return extracted_text


async def refine_prompt(system_prompt: str, cache_dir: Path = None) -> str:
    """Strips rigid legacy formatting constraints from the Master Prompt."""
    if cache_dir:
        prompt_hash = hashlib.md5(system_prompt.encode("utf-8")).hexdigest()
        cache_file = (
            cache_dir / f"refined_prompt_{settings.refinement_model}_{prompt_hash}.md"
        )
        if cache_file.exists():
            logger.info(
                f"Using cached refined prompt using {settings.refinement_model}"
            )
            return cache_file.read_text(encoding="utf-8")

    logger.info(f"Cleaning legacy constraints using {settings.refinement_model}")

    refine_prompt_instructions = f"""
    You are an expert prompt engineer. Your task is to clean up the following system prompt.
    The user might have included strict formatting instructions like "Only output a CSV" or "Format your answer as a table".
    Strip out ALL instructions related to output formatting, syntax, or data structure. 
    Keep all instructions related to the analytical task, definitions, and criteria.
    
    ORIGINAL PROMPT:
    {system_prompt}
    
    Return ONLY the cleaned prompt text, nothing else.
    """

    client = get_client()
    response = await client.aio.models.generate_content(
        model=settings.refinement_model,
        contents=refine_prompt_instructions,
        config=types.GenerateContentConfig(temperature=0.1),
    )

    clean_prompt = response.text
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(clean_prompt, encoding="utf-8")
    return clean_prompt


async def upload_pdf(pdf_path: str) -> Any:
    """Uploads a PDF file using the GenAI File API and returns the File object."""
    logger.info(f"Uploading PDF {pdf_path} using File API")
    client = get_client()
    uploaded_file = await client.aio.files.upload(file=pdf_path)
    return uploaded_file


async def delete_pdf(uploaded_file: Any):
    """Deletes an uploaded PDF file."""
    if uploaded_file and hasattr(uploaded_file, "name"):
        logger.info(f"Deleting uploaded PDF {uploaded_file.name}")
        client = get_client()
        await client.aio.files.delete(name=uploaded_file.name)


def _get_document_part(document_context: Union[str, Any]) -> types.Part:
    """Helper to format the document context appropriately for the API."""
    if isinstance(document_context, str):
        return types.Part.from_text(
            text=f"RESEARCH PAPER MARKDOWN:\n\n{document_context}"
        )
    else:
        # It's an uploaded File object
        return document_context


async def run_fast_mode(
    document_context: Union[str, Any],
    clean_prompt: str,
    outdir: Path = None,
    with_justification: bool = True,
    expected_count: int = 0,
) -> AssessmentReport:
    """Executes a single-shot fast evaluation async."""

    logger.info(f"Running fast evaluation using {settings.reasoning_model}")
    client = get_client()

    report = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        response = await client.aio.models.generate_content(
            model=settings.reasoning_model,
            contents=[
                types.Part.from_text(text=f"Master Criteria:\n{clean_prompt}"),
                _get_document_part(document_context),
            ],
            config=types.GenerateContentConfig(
                temperature=settings.temperature,
                response_mime_type="application/json",
                response_schema=AssessmentReport,
            ),
        )
        try:
            report = AssessmentReport.model_validate_json(response.text)
            if expected_count > 0 and len(report.answers) < expected_count:
                logger.warning(
                    f"Attempt {attempt + 1}: Model returned {len(report.answers)} answers, expected {expected_count}. Retrying..."
                )
                if attempt < max_retries:
                    continue
            break
        except Exception as e:
            text_snippet = (
                response.text[:100] if hasattr(response, "text") else "NO TEXT"
            )
            logger.warning(
                f"Attempt {attempt + 1}: Failed to parse JSON: {e}. Snippet: {text_snippet}. Retrying..."
            )
            if attempt >= max_retries:
                raise e

    # Post-process to remove justification if not requested
    if not with_justification:
        for ans in report.answers:
            ans.justification = None

    if outdir:
        (outdir / "fast_report_artifact.json").write_text(
            report.model_dump_json(indent=4), encoding="utf-8"
        )

    return report


async def extract_evidence_batch(client, group, document_context) -> EvidenceArtifact:
    evidence_prompt = f"""
    You are analyzing the following task group: {group.group_name}.
    Questions in this group:
    {[sub.question_text for sub in group.subtasks]}
    
    Scan the provided document context and extract exact quotes that provide evidence to answer these questions.
    """

    contents = [
        types.Part.from_text(text=evidence_prompt),
        _get_document_part(document_context),
    ]

    evidence_artifact = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        evidence_resp = await client.aio.models.generate_content(
            model=settings.reasoning_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=settings.temperature,
                response_mime_type="application/json",
                response_schema=EvidenceArtifact,
            ),
        )
        try:
            evidence_artifact = EvidenceArtifact.model_validate_json(evidence_resp.text)
            break
        except Exception as e:
            text_snippet = (
                evidence_resp.text[:100]
                if hasattr(evidence_resp, "text")
                else "NO TEXT"
            )
            logger.warning(
                f"Attempt {attempt + 1}: Failed to parse Evidence JSON: {e}. Snippet: {text_snippet}. Retrying..."
            )
            if attempt >= max_retries:
                raise e
    return evidence_artifact


async def run_planning_mode_concurrent(
    document_context: Union[str, Any],
    clean_prompt: str,
    outdir: Path = None,
    with_justification: bool = True,
    expected_count: int = 0,
) -> AssessmentReport:
    """Executes the methodical Planning mode async using TaskGroups and Evidence extraction."""

    logger.info(f"Running planning mode using {settings.reasoning_model}")

    client = get_client()

    # 1. Initialize Task List
    logger.info("Planning Mode: Task Generation")
    init_prompt = f"""
    You are an AI tasked with evaluating a research paper based on the following master prompt criteria:
    {clean_prompt}
    
    Break down these questions into logical Task Groups (e.g., 'Sample Size', 'Blinding', 'Intervention').
    Generate a complete TaskListArtifact based on the questions provided in the criteria.
    """

    task_list_response = await client.aio.models.generate_content(
        model=settings.reasoning_model,
        contents=init_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=TaskListArtifact,
        ),
    )

    task_list = TaskListArtifact.model_validate_json(task_list_response.text)

    if outdir:
        (outdir / "plan_tasks.json").write_text(task_list.model_dump_json(indent=4))

    # 2. Extract Evidence per Task Group concurrently
    logger.info("Planning Mode: Evidence Extraction (Concurrent)")

    evidence_tasks = [
        extract_evidence_batch(client, group, document_context)
        for group in task_list.groups
    ]
    evidence_artifacts = await asyncio.gather(*evidence_tasks)

    if outdir:
        (outdir / "plan_evidence.json").write_text(
            json.dumps([e.model_dump() for e in evidence_artifacts], indent=4)
        )

    # 3. Assess and Synthesize Final Report
    logger.info("Planning Mode: Final Synthesis")
    synthesis_prompt = f"""
    You are evaluating the final Assessment Report based on the following master criteria:
    {clean_prompt}
    
    Below is the extracted evidence directly sourced from the paper:
    {json.dumps([ev.model_dump() for ev in evidence_artifacts], indent=4)}
    
    Based strictly on this evidence (to mitigate affirmative bias), answer each question with `True` (Yes/Practiced) or `False` (No/Not Practiced).
    Provide a short justification based ONLY on the evidence.
    """

    report = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        final_response = await client.aio.models.generate_content(
            model=settings.reasoning_model,
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature,
                response_mime_type="application/json",
                response_schema=AssessmentReport,
            ),
        )
        try:
            report = AssessmentReport.model_validate_json(final_response.text)
            if expected_count > 0 and len(report.answers) < expected_count:
                logger.warning(
                    f"Attempt {attempt + 1}: Model returned {len(report.answers)} answers, expected {expected_count}. Retrying..."
                )
                if attempt < max_retries:
                    continue
            break
        except Exception as e:
            text_snippet = (
                final_response.text[:100]
                if hasattr(final_response, "text")
                else "NO TEXT"
            )
            logger.warning(
                f"Attempt {attempt + 1}: Failed to parse JSON: {e}. Snippet: {text_snippet}. Retrying..."
            )
            if attempt >= max_retries:
                raise e

    if not with_justification:
        for ans in report.answers:
            ans.justification = None

    if outdir:
        (outdir / "plan_report_artifact.json").write_text(
            report.model_dump_json(indent=4)
        )

    return report


async def run_planning_mode_sequential(
    document_context: Union[str, Any],
    clean_prompt: str,
    outdir: Path = None,
    with_justification: bool = True,
    expected_count: int = 0
) -> AssessmentReport:
    """Executes the Planning mode using a sequential ChatSession to save tokens and maintain state."""

    logger.info(f"Running sequential planning mode using {settings.reasoning_model}")
    client = get_client()

    # 1. Initialize Task List
    logger.info("Sequential Planning Mode: Task Generation")
    init_prompt = f"""
    You are an AI tasked with evaluating a research paper based on the following master prompt criteria:
    {clean_prompt}
    
    Break down these questions into logical Task Groups (e.g., 'Sample Size', 'Blinding', 'Intervention').
    Generate a complete TaskListArtifact based on the questions provided in the criteria.
    """

    task_list_response = await client.aio.models.generate_content(
        model=settings.reasoning_model,
        contents=init_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=TaskListArtifact,
        ),
    )

    task_list = TaskListArtifact.model_validate_json(task_list_response.text)
    if outdir:
        (outdir / "plan_seq_tasks.json").write_text(task_list.model_dump_json(indent=4))

    # 2. Sequential Evidence Extraction via ChatSession
    logger.info("Sequential Planning Mode: Evidence Extraction (ChatSession)")

    # Start chat with the document as the initial context
    chat = client.aio.chats.create(
        model=settings.reasoning_model,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            system_instruction="You are an expert research paper analyst extracting evidence.",
        ),
    )

    # We must explicitly send the document first
    await chat.send_message(
        [
            types.Part.from_text(
                text="Please read and remember the following document for subsequent questions."
            ),
            _get_document_part(document_context),
        ]
    )

    evidence_artifacts = []
    for group in task_list.groups:
        logger.info(f"Extracting evidence for group: {group.group_name}")
        evidence_prompt = f"""
        You are analyzing the following task group: {group.group_name}.
        Questions in this group:
        {[sub.question_text for sub in group.subtasks]}
        
        Scan the document and extract exact quotes that provide evidence to answer these questions.
        If there is no explicit evidence, state that this is missing in the source context.
        Return an EvidenceArtifact.
        """

        # We override response schema for this specific turn
        evidence_resp = await chat.send_message(
            message=evidence_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=EvidenceArtifact
            ),
        )
        evidence_artifacts.append(
            EvidenceArtifact.model_validate_json(evidence_resp.text)
        )

    if outdir:
        (outdir / "plan_seq_evidence.json").write_text(
            json.dumps([e.model_dump() for e in evidence_artifacts], indent=4)
        )

    # 3. Assess and Synthesize Final Report
    logger.info("Sequential Planning Mode: Final Synthesis")
    synthesis_prompt = f"""
    You are evaluating the final Assessment Report based on the following master criteria:
    {clean_prompt}
    
    Below is the extracted evidence directly sourced from the paper:
    {json.dumps([ev.model_dump() for ev in evidence_artifacts], indent=4)}
    
    Based strictly on this evidence (to mitigate affirmative bias), answer each question with `True` (Yes/Practiced) or `False` (No/Not Practiced).
    Provide a short justification based ONLY on the evidence.
    """

    report = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        final_response = await client.aio.models.generate_content(
            model=settings.reasoning_model,
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature,
                response_mime_type="application/json",
                response_schema=AssessmentReport,
            ),
        )
        try:
            report = AssessmentReport.model_validate_json(final_response.text)
            if expected_count > 0 and len(report.answers) < expected_count:
                logger.warning(
                    f"Attempt {attempt + 1}: Model returned {len(report.answers)} answers, expected {expected_count}. Retrying..."
                )
                if attempt < max_retries:
                    continue
            break
        except Exception as e:
            text_snippet = (
                final_response.text[:100]
                if hasattr(final_response, "text")
                else "NO TEXT"
            )
            logger.warning(
                f"Attempt {attempt + 1}: Failed to parse JSON: {e}. Snippet: {text_snippet}. Retrying..."
            )
            if attempt >= max_retries:
                raise e

    if not with_justification:
        for ans in report.answers:
            ans.justification = None

    if outdir:
        (outdir / "plan_seq_report_artifact.json").write_text(
            report.model_dump_json(indent=4)
        )

    return report


async def analyze_mismatches(
    document_context: Union[str, Any],
    mismatch_details: List[EvaluationDetail],
    expected_csv_prompt: str,
) -> MismatchReport:
    """Diagnoses why the model predictions did not match the expected ground truth."""
    if not mismatch_details:
        return MismatchReport(analyses=[])

    logger.info(
        f"Running Mismatch Analyzer for {len(mismatch_details)} incorrect answers using {settings.mismatch_analysis_model}"
    )
    client = get_client()

    mismatch_prompt = f"""
    You are an AI diagnostic expert. A previous evaluation model made errors when analyzing a research paper.
    Your task is to analyze WHY the model failed for specific questions.
    
    Original Evaluation Criteria Overview:
    {expected_csv_prompt}
    
    Here are the specific questions it got wrong, including the expected answer and what the model predicted (with its justification):
    {[d.model_dump_json() for d in mismatch_details]}
    
    Analyze each error against the source document. Output a MismatchReport containing your analysis for each mismatched question.
    Reasons for model errors can include but are not limited to: 'Model hallucinated/failed', 'Ambiguous Prompt', 'Better than Ground Truth', etc.
    """

    contents = [
        types.Part.from_text(text=mismatch_prompt),
        _get_document_part(document_context),
    ]

    report = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        response = await client.aio.models.generate_content(
            model=settings.mismatch_analysis_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,  # Low temperature for analytical diagnostic
                response_mime_type="application/json",
                response_schema=MismatchReport,
            ),
        )
        try:
            report = MismatchReport.model_validate_json(response.text)
            break
        except Exception as e:
            text_snippet = (
                response.text[:100] if hasattr(response, "text") else "NO TEXT"
            )
            logger.warning(
                f"Attempt {attempt + 1}: Failed to parse Mismatch JSON: {e}. Snippet: {text_snippet}. Retrying..."
            )
            if attempt >= max_retries:
                raise e

    return report
