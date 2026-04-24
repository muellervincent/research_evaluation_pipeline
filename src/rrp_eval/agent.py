import asyncio
from pathlib import Path
from google import genai
from google.genai import types
from loguru import logger
import json

from .config import settings
from .schema import (
    TaskListArtifact, EvidenceArtifact, AssessmentReport
)

# Async helper to get client per call to respect active settings (since they can switch)
def get_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)

async def process_pdf(pdf_path: str, cache_dir: Path = None) -> str:
    """Pre-processes a PDF, extracting exact text and describing images."""
    if cache_dir:
        stem = Path(pdf_path).stem
        cache_file = cache_dir / f"{stem}.md"
        if cache_file.exists():
            logger.info(f"Using cached markdown for {stem}")
            return cache_file.read_text(encoding="utf-8")
            
    logger.info(f"Extracting text & image descriptions using {settings.extraction_model}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    extraction_prompt = """
    Extract all the text from this document exactly as it is, without summarization.
    For any images, charts, or graphs present in the PDF, provide a plain text description 
    of the visual information without over-interpreting it. Format the overall output in Markdown.
    """
    
    client = get_client()
    response = await client.aio.models.generate_content(
        model=settings.extraction_model,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=extraction_prompt)
        ],
        config=types.GenerateContentConfig(temperature=0.1)
    )
    
    extracted_text = response.text
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(pdf_path).stem
        cache_file = cache_dir / f"{stem}.md"
        cache_file.write_text(extracted_text, encoding="utf-8")
        
    return extracted_text

async def refine_prompt(system_prompt: str) -> str:
    """Strips rigid legacy formatting constraints from the Master Prompt."""
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
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text

async def run_fast_mode(markdown_text: str, system_prompt: str, outdir: Path = None, with_justification: bool = True) -> AssessmentReport:
    """Executes a single-shot fast evaluation async."""
    clean_prompt = await refine_prompt(system_prompt)
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "fast_cleaned_prompt.txt").write_text(clean_prompt, encoding="utf-8")

    logger.info(f"Running fast evaluation using {settings.reasoning_model}")
    
    client = get_client()
    response = await client.aio.models.generate_content(
        model=settings.reasoning_model,
        contents=[
            types.Part.from_text(text=f"Master Criteria:\n{clean_prompt}"),
            types.Part.from_text(text=f"RESEARCH PAPER MARKDOWN:\n\n{markdown_text}")
        ],
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=AssessmentReport
        )
    )
    
    report = AssessmentReport.model_validate_json(response.text)
    
    # Post-process to remove justification if not requested
    if not with_justification:
        for ans in report.answers:
            ans.justification = None
            
    if outdir:
        (outdir / "fast_report_artifact.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            
    return report

async def extract_evidence_batch(client, group, markdown_text) -> EvidenceArtifact:
    evidence_prompt = f"""
    You are analyzing the following task group: {group.group_name}.
    Questions in this group:
    {[sub.question_text for sub in group.subtasks]}
    
    Scan the following research paper markdown and extract exact quotes that provide evidence to answer these questions.
    If there is no explicit evidence, state that this is missing in the source context.
    Return an EvidenceArtifact.
    
    RESEARCH PAPER MARKDOWN:
    {markdown_text}
    """
    
    evidence_resp = await client.aio.models.generate_content(
        model=settings.reasoning_model,
        contents=evidence_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=EvidenceArtifact
        )
    )
    return EvidenceArtifact.model_validate_json(evidence_resp.text)

async def run_planning_mode(markdown_text: str, system_prompt: str, outdir: Path = None, with_justification: bool = True) -> AssessmentReport:
    """Executes the methodical Planning mode async using TaskGroups and Evidence extraction."""
    clean_prompt = await refine_prompt(system_prompt)
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "plan_cleaned_prompt.txt").write_text(clean_prompt, encoding="utf-8")
        
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
            response_schema=TaskListArtifact
        )
    )
    
    task_list = TaskListArtifact.model_validate_json(task_list_response.text)
    
    if outdir:
        (outdir / "plan_tasks.json").write_text(task_list.model_dump_json(indent=2))

    # 2. Extract Evidence per Task Group concurrently
    logger.info("Planning Mode: Evidence Extraction (Concurrent)")
    
    evidence_tasks = [extract_evidence_batch(client, group, markdown_text) for group in task_list.groups]
    evidence_artifacts = await asyncio.gather(*evidence_tasks)
            
    if outdir:
        (outdir / "plan_evidence.json").write_text(json.dumps([e.model_dump() for e in evidence_artifacts], indent=2))

    # 3. Assess and Synthesize Final Report
    logger.info("Planning Mode: Final Synthesis")
    synthesis_prompt = f"""
    You are evaluating the final Assessment Report based on the following master criteria:
    {clean_prompt}
    
    Below is the extracted evidence directly sourced from the paper:
    {[ev.model_dump_json() for ev in evidence_artifacts]}
    
    Based strictly on this evidence (to mitigate affirmative bias), answer each question with `True` (Yes/Practiced) or `False` (No/Not Practiced).
    Provide a short justification based ONLY on the evidence.
    """
    
    final_response = await client.aio.models.generate_content(
        model=settings.reasoning_model,
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=AssessmentReport
        )
    )
    
    report = AssessmentReport.model_validate_json(final_response.text)
    
    if not with_justification:
        for ans in report.answers:
            ans.justification = None
            
    if outdir:
        (outdir / "plan_report_artifact.json").write_text(report.model_dump_json(indent=2))
        
    return report
