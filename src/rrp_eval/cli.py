from rrp_eval.agent import refine_prompt
import asyncio
import json
import typer
from pathlib import Path
from datetime import datetime
import random
from typing import Optional

from .config import config, settings, load_prompt
from .logger import setup_logging, highlight_print
from .agent import process_pdf, upload_pdf, delete_pdf, run_fast_mode, run_planning_mode_concurrent, run_planning_mode_sequential, analyze_mismatches
from .evaluate import load_expected_answers, evaluate_report, compare_results
from .schema import EvaluationResult, EvaluationMetrics

app = typer.Typer(help="AI-Assisted RRP Analysis Tool")

def get_pdfs(target_path: Path, sample_size: Optional[int] = None, subset: Optional[str] = None) -> list[Path]:
    if target_path.is_file():
        return [target_path]
    elif target_path.is_dir():
        pdfs = list(target_path.glob("*.pdf"))
        pdfs.sort()
        if subset:
            allowed_stems = [s.strip() for s in subset.split(",")]
            pdfs = [p for p in pdfs if p.stem in allowed_stems]
        elif sample_size and sample_size < len(pdfs):
            return random.sample(pdfs, sample_size)
        return pdfs
    return []

async def process_pdf_workflow(
    pdf_path: Path, 
    mode: str, 
    prompt_text: str, 
    expected_csv: str, 
    outdir: Path,
    with_justification: bool,
    cache_dir: Path,
    ingestion: str,
    planning_arch: str,
    analyze_mismatches_flag: bool,
) -> Optional[dict]:
    pdf_stem = pdf_path.stem
    highlight_print(f"--- Processing {pdf_stem} [Mode: {mode.upper()}, Ingestion: {ingestion.upper()}] ---")
    
    pdf_outdir = outdir / pdf_stem
    artifacts_dir = pdf_outdir / "artifacts"
    results_dir = pdf_outdir / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    document_context = None
    try:
        if ingestion == "pdf":
            document_context = await upload_pdf(str(pdf_path))
        else:
            document_context = await process_pdf(str(pdf_path), cache_dir=cache_dir)
    except Exception as e:
        highlight_print(f"Error ingesting PDF {pdf_path}: {e}")
        return None

    try:
        clean_prompt = await refine_prompt(prompt_text, cache_dir)
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (outdir / "cleaned_prompt.txt").write_text(clean_prompt, encoding="utf-8")
    except Exception as e:
        highlight_print(f"Error refining prompt: {e}")
        return None
        
    expected = load_expected_answers(expected_csv, pdf_stem)
    results = {}
    
    async def run_and_eval(run_mode: str):
        eval_result_obj = EvaluationResult(
            mode=run_mode,
            pdf_stem=pdf_stem,
            model_name=settings.reasoning_model,
        )
        try:
            if run_mode == "fast":
                result_artifact = await run_fast_mode(document_context, clean_prompt, outdir=artifacts_dir, with_justification=with_justification, expected_count=len(expected))
            elif run_mode == "planning":
                if planning_arch == "sequential":
                    result_artifact = await run_planning_mode_sequential(
                        document_context, clean_prompt, outdir=artifacts_dir, with_justification=with_justification, expected_count=len(expected)
                    )
                else:
                    result_artifact = await run_planning_mode_concurrent(
                        document_context, clean_prompt, outdir=artifacts_dir, with_justification=with_justification, expected_count=len(expected)
                    )
            
            eval_result_obj.raw_output = result_artifact.model_dump()
            metrics_dict = evaluate_report(eval_result_obj.raw_output, expected)
            eval_result_obj.metrics = EvaluationMetrics(**metrics_dict)
            
            if analyze_mismatches_flag and eval_result_obj.metrics.details:
                incorrect_details = [d for d in eval_result_obj.metrics.details if not d.correct]
                if incorrect_details:
                    mismatch_report = await analyze_mismatches(document_context, incorrect_details, prompt_text)
                    
                    # Merge analysis into details
                    analysis_map = {a.question_number: (a.explanation, a.category) for a in mismatch_report.analyses}
                    for d in eval_result_obj.metrics.details:
                        if d.question_number in analysis_map:
                            d.mismatch_reason = analysis_map[d.question_number][0]
                            d.mismatch_category = analysis_map[d.question_number][1]
                    
                    (artifacts_dir / f"{run_mode}_mismatch_analysis.json").write_text(mismatch_report.model_dump_json(indent=4), encoding="utf-8")
            
            acc = eval_result_obj.metrics.accuracy
            corr = eval_result_obj.metrics.correct
            tot = eval_result_obj.metrics.total
            missing = eval_result_obj.metrics.missing_answers
            warning_msg = f" [WARNING: {missing} Missing Answers!]" if missing > 0 else ""
            highlight_print(f"Outcome ({run_mode.upper()}) => Accuracy: {acc*100:.2f}% ({corr}/{tot}){warning_msg}")
            
            # Save raw output
            raw_file = artifacts_dir / f"{run_mode}_raw.json"
            raw_file.write_text(json.dumps(eval_result_obj.raw_output, indent=4), encoding="utf-8")
            
            # Save evaluation results
            eval_file = results_dir / f"{run_mode}_eval.json"
            if eval_result_obj.metrics:
                eval_file.write_text(eval_result_obj.metrics.model_dump_json(indent=4), encoding="utf-8")
            
            results[run_mode] = eval_file
            
        except Exception as e:
            highlight_print(f"Execution or evaluation failed for {run_mode}: {e}")

    if mode in ("fast", "both"):
        await run_and_eval("fast")
    if mode in ("planning", "both"):
        await run_and_eval("planning")
        
    if ingestion == "pdf" and document_context:
        await delete_pdf(document_context)
        
    return results

@app.command()
def evaluate(
    target: Path = typer.Argument(..., help="Path to the research paper PDF file or directory"),
    sample_size: Optional[int] = typer.Option(None, help="Number of papers to randomly sample"),
    subset: Optional[str] = typer.Option(None, help="Comma separated list of PDF stems to evaluate"),
    mode: str = typer.Option("fast", help="Execution mode ('fast', 'planning', 'both')"),
    ingestion: str = typer.Option("extraction", help="Ingestion pathway ('extraction', 'pdf')"),
    planning_arch: str = typer.Option("concurrent", help="Planning architecture ('concurrent', 'sequential')"),
    profile: str = typer.Option("unpaid_lite", help="Configuration profile to use from eval_profiles.toml"),
    with_justification: bool = typer.Option(True, help="Whether to include justifications in the output"),
    analyze_mismatches_flag: bool = typer.Option(False, "--analyze-mismatches", help="Run mismatch analyzer on incorrect predictions"),
    prompt: Path = typer.Option(Path("resources/prompt_detailed.md"), help="Path to master prompt file"),
    expected: str = typer.Option("resources/csv/correct_answers.csv", help="Path to ground truth CSV"),
    outdir: Path = typer.Option(Path("./output"), help="Directory to save outputs")
):
    """Run evaluation on PDFs against defined criteria."""
    config.set_profile(profile)
    
    pdfs = get_pdfs(target, sample_size, subset)
    if not pdfs:
        typer.echo("No PDFs found to process.")
        raise typer.Exit()
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_outdir = outdir / timestamp
    run_outdir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path("resources/preprocessing_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(run_outdir / "run.log")
    prompt_text = load_prompt(str(prompt))
    
    async def run_all():
        tasks = []
        for pdf in pdfs:
            tasks.append(
                process_pdf_workflow(
                    pdf, mode, prompt_text, expected, run_outdir, with_justification, cache_dir, ingestion, planning_arch, analyze_mismatches_flag
                )
            )
                
        all_results = await asyncio.gather(*tasks)
        
        if mode == "both":
            for i, pdf in enumerate(pdfs):
                res = all_results[i]
                if res and "fast" in res and "planning" in res:
                    report = compare_results(str(res["fast"]), str(res["planning"]))
                    highlight_print(f"--- Comparison Report for {pdf.stem} ---")
                    typer.echo(report)
                    (run_outdir / pdf.stem / "results" / f"{pdf.stem}_comparison.md").write_text(report, encoding="utf-8")
                    
    asyncio.run(run_all())

@app.command()
def compare(
    file1: Path = typer.Argument(..., help="First JSON result file"),
    file2: Path = typer.Argument(..., help="Second JSON result file")
):
    """Compare two JSON result files directly."""
    report = compare_results(str(file1), str(file2))
    typer.echo(report)

if __name__ == "__main__":
    app()
