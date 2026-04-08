from google import genai
from google.genai import types
import pymupdf4llm
from config import settings
from schema import TaskListArtifact, EvidenceArtifact, AssessmentReportArtifact

client = genai.Client(api_key=settings.gemini_api_key)

def process_pdf(pdf_path: str) -> str:
    """Converts a PDF to Markdown representation using pymupdf4llm."""
    return pymupdf4llm.to_markdown(pdf_path)

def run_fast_mode(markdown_text: str, system_prompt: str) -> str:
    """
    Executes a single-shot fast evaluation where the prompt requests a CSV response.
    """
    response = client.models.generate_content(
        model=settings.model,
        contents=[
            types.Part.from_text(text=system_prompt),
            types.Part.from_text(text=f"RESEARCH PAPER MARKDOWN:\n\n{markdown_text}")
        ],
        config=types.GenerateContentConfig(
            temperature=settings.temperature
        )
    )
    return response.text

def run_planning_mode(markdown_text: str, system_prompt: str) -> AssessmentReportArtifact:
    """
    Executes the methodical Planning mode evaluation using TaskGroups and Evidence extraction.
    """
    # 1. Initialize Task List
    init_prompt = f"""
    You are an AI tasked with evaluating a research paper based on the following master prompt criteria:
    {system_prompt}
    
    Please break down these questions into logical Task Groups (e.g., 'Sample Size', 'Blinding', 'Intervention').
    Generate a complete TaskListArtifact based on the questions provided in the criteria.
    """
    
    task_list_response = client.models.generate_content(
        model=settings.model,
        contents=init_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=TaskListArtifact
        )
    )
    
    task_list = TaskListArtifact.model_validate_json(task_list_response.text)
    
    # 2. Extract Evidence per Task Group
    evidence_artifacts = []
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
            model=settings.model,
            contents=evidence_prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature,
                response_mime_type="application/json",
                response_schema=EvidenceArtifact
            )
        )
        evidence_artifacts.append(EvidenceArtifact.model_validate_json(evidence_resp.text))

    # 3. Assess and Synthesize Final Report
    synthesis_prompt = f"""
    You are evaluating the final Assessment Report based on the following master criteria:
    {system_prompt}
    
    Below is the extracted evidence directly sourced from the paper:
    {[ev.model_dump_json() for ev in evidence_artifacts]}
    
    Based strictly on this evidence (to mitigate affirmative bias), answer each question with a definitive Yes or No and provide a short justification based ONLY on the evidence.
    """
    
    final_response = client.models.generate_content(
        model=settings.model,
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(
            temperature=settings.temperature,
            response_mime_type="application/json",
            response_schema=AssessmentReportArtifact
        )
    )
    
    return AssessmentReportArtifact.model_validate_json(final_response.text)
