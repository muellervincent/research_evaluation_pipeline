from pathlib import Path
from google import genai
from google.genai import types
import pymupdf4llm
from config import settings
from schema import TaskListArtifact, EvidenceArtifact, AssessmentReportArtifact, FastAssessmentResult
from logger import print_cognitive_step, highlight_print
from rich.prompt import Confirm
import json

client = genai.Client(api_key=settings.gemini_api_key)

def process_pdf(pdf_path: str) -> str:
    """Pre-processes a PDF, extracting exact text and describing images using the extraction model."""
    print_cognitive_step("Pre-processing PDF", f"Extracting text & image descriptions using {settings.extraction_model}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    extraction_prompt = """
    Extract all the text from this document exactly as it is, without summarization.
    For any images, charts, or graphs present in the PDF, provide a plain text description 
    of the visual information without over-interpreting it. Format the overall output in Markdown.
    """
    
    response = client.models.generate_content(
        model=settings.extraction_model,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=extraction_prompt)
        ],
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return response.text

def refine_prompt(system_prompt: str) -> str:
    """Strips rigid legacy formatting constraints (like forcing CSV) from the Master Prompt."""
    print_cognitive_step("Refining Prompt", f"Cleaning legacy constraints using {settings.refinement_model}")
    
    refine_prompt_instructions = f"""
    You are an expert prompt engineer. Your task is to clean up the following system prompt.
    The user might have included strict formatting instructions like "Only output a CSV" or "Format your answer as a table".
    Strip out ALL instructions related to output formatting, syntax, or data structure. 
    Keep all instructions related to the analytical task, definitions, and criteria.
    
    ORIGINAL PROMPT:
    {system_prompt}
    
    Return ONLY the cleaned prompt text, nothing else.
    """
    
    response = client.models.generate_content(
        model=settings.refinement_model,
        contents=refine_prompt_instructions,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return response.text

def run_fast_mode(markdown_text: str, system_prompt: str, interactive: bool = False, strategy: str = "batch") -> FastAssessmentResult:
    """Executes a single-shot fast evaluation."""
    clean_prompt = refine_prompt(system_prompt)
    
    if interactive:
        highlight_print("--- Interactive Mode: Fast Iteration ---")
        print_cognitive_step("Cleaned Prompt", clean_prompt)
        if not Confirm.ask("Do you want to proceed with this cleaned prompt?", default=True):
            highlight_print("Run aborted by user.")
            raise InterruptedError("User aborted after prompt refinement.")
            
    # For fast mode, the batch strategy is the native approach (checking all at once).
    # If strategy is "isolated", we would run it question by question (requires parsing prompt).
    # For now, fast mode is assumed to be batch.
    print_cognitive_step("Fast Evaluation", f"Running fast evaluation using {settings.reasoning_model} (Strategy: {strategy})")
    
    response = client.models.generate_content(
        model=settings.reasoning_model,
        contents=[
            types.Part.from_text(text=f"Master Criteria:\n{clean_prompt}"),
            types.Part.from_text(text=f"RESEARCH PAPER MARKDOWN:\n\n{markdown_text}")
        ],
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=FastAssessmentResult
        )
    )
    return FastAssessmentResult.model_validate_json(response.text)

def run_planning_mode(markdown_text: str, system_prompt: str, interactive: bool = False, strategy: str = "batch", outdir: Path = None) -> AssessmentReportArtifact:
    """Executes the methodical Planning mode evaluation using TaskGroups and Evidence extraction."""
    clean_prompt = refine_prompt(system_prompt)
    
    # 1. Initialize Task List
    print_cognitive_step("Planning Mode: Task Generation", f"Generating task list using {settings.reasoning_model}")
    init_prompt = f"""
    You are an AI tasked with evaluating a research paper based on the following master prompt criteria:
    {clean_prompt}
    
    Break down these questions into logical Task Groups (e.g., 'Sample Size', 'Blinding', 'Intervention').
    Generate a complete TaskListArtifact based on the questions provided in the criteria.
    """
    
    task_list_response = client.models.generate_content(
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
        
    if interactive:
        highlight_print("--- Interactive Mode: Planning Phase ---")
        print_cognitive_step("Generated Task List", task_list.model_dump_json(indent=2))
        if not Confirm.ask("Do you approve this task extraction plan?", default=True):
            raise InterruptedError("User rejected the task plan.")

    # 2. Extract Evidence per Task Group/Task
    print_cognitive_step("Planning Mode: Evidence Extraction", f"Extracting evidence (Strategy: {strategy})")
    evidence_artifacts = []
    
    if strategy == "isolated":
        # Loop through each subtask independently for maximum context focus
        for group in task_list.groups:
            group_evidence = []
            for subtask in group.subtasks:
                evidence_prompt = f"""
                You are isolating evidence for the following specific question in the group '{group.group_name}':
                Question {subtask.question_number}: {subtask.question_text}
                
                Scan the following research paper markdown and extract exact quotes that provide evidence to answer this question.
                If there is no explicit evidence, state that this is missing in the source context.
                Return an EvidenceArtifact containing just this item.
                
                RESEARCH PAPER MARKDOWN:
                {markdown_text}
                """
                evidence_resp = client.models.generate_content(
                    model=settings.reasoning_model,
                    contents=evidence_prompt,
                    config=types.GenerateContentConfig(
                        temperature=settings.temperature,
                        response_mime_type="application/json",
                        response_schema=EvidenceArtifact
                    )
                )
                evidence_artifacts.append(EvidenceArtifact.model_validate_json(evidence_resp.text))
    else:
        # Default Strategy: Batch (Summed up by group)
        for group in task_list.groups:
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
            
            evidence_resp = client.models.generate_content(
                model=settings.reasoning_model,
                contents=evidence_prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.temperature,
                    response_mime_type="application/json",
                    response_schema=EvidenceArtifact
                )
            )
            evidence_artifacts.append(EvidenceArtifact.model_validate_json(evidence_resp.text))
            
    if outdir:
        (outdir / "plan_evidence.json").write_text(json.dumps([e.model_dump() for e in evidence_artifacts], indent=2))

    # 3. Assess and Synthesize Final Report
    print_cognitive_step("Planning Mode: Final Synthesis", "Generating final Assessment Report")
    synthesis_prompt = f"""
    You are evaluating the final Assessment Report based on the following master criteria:
    {clean_prompt}
    
    Below is the extracted evidence directly sourced from the paper:
    {[ev.model_dump_json() for ev in evidence_artifacts]}
    
    Based strictly on this evidence (to mitigate affirmative bias), answer each question with `True` (Yes/Practiced) or `False` (No/Not Practiced) and provide a short justification based ONLY on the evidence.
    """
    
    final_response = client.models.generate_content(
        model=settings.reasoning_model,
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=AssessmentReportArtifact
        )
    )
    
    report = AssessmentReportArtifact.model_validate_json(final_response.text)
    
    if outdir:
        (outdir / "plan_report_artifact.json").write_text(report.model_dump_json(indent=2))
        
    return report
