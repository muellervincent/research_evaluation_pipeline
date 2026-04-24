import asyncio
import json
import typer
from pathlib import Path
from datetime import datetime
import random
from typing import Optional

from .config import config, settings, load_prompt
from .logger import setup_logging, highlight_print
from .agent import process_pdf, run_fast_mode, run_planning_mode
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

async def process_single_pdf(
    pdf_path: Path, 
    mode: str, 
    prompt_text: str, 
    expected_csv: str, 
    outdir: Path,
    with_justification: bool,
    cache_dir: Path
) -> Optional[Path]:
    pdf_stem = pdf_path.stem
    highlight_print(f"--- Processing {pdf_stem} [{mode.upper()} mode] ---")
    
    try:
        markdown_text = await process_pdf(str(pdf_path), cache_dir=cache_dir)
    except Exception as e:
        highlight_print(f"Error extracting PDF {pdf_path}: {e}")
        return None
        
    eval_result_obj = EvaluationResult(
        mode=mode,
        pdf_stem=pdf_stem,
        model_name=settings.reasoning_model,
    )
    
    expected = load_expected_answers(expected_csv, pdf_stem)
    
    try:
        if mode == "fast":
            result_artifact = await run_fast_mode(markdown_text, prompt_text, outdir=outdir, with_justification=with_justification)
            eval_result_obj.raw_output = result_artifact.model_dump()
            metrics_dict = evaluate_report(eval_result_obj.raw_output, expected)
            
        elif mode == "planning":
            result_artifact = await run_planning_mode(
                markdown_text, 
                prompt_text, 
                outdir=outdir,
                with_justification=with_justification
            )
            eval_result_obj.raw_output = result_artifact.model_dump()
            metrics_dict = evaluate_report(eval_result_obj.raw_output, expected)
            
        eval_result_obj.metrics = EvaluationMetrics(**metrics_dict)
        acc = metrics_dict.get('accuracy', 0)
        corr = metrics_dict.get('correct', 0)
        tot = metrics_dict.get('total', 0)
        highlight_print(f"Outcome => Accuracy: {acc*100:.2f}% ({corr}/{tot})")
        
    except Exception as e:
        highlight_print(f"Execution or evaluation failed: {e}")
        return None
        
    # Save raw output
    raw_file = outdir / f"{pdf_stem}_{mode}_raw.json"
    raw_file.write_text(json.dumps(eval_result_obj.raw_output, indent=2), encoding="utf-8")
    
    # Save evaluation results
    eval_file = outdir / f"{pdf_stem}_{mode}_eval.json"
    if eval_result_obj.metrics:
        eval_file.write_text(eval_result_obj.metrics.model_dump_json(indent=2), encoding="utf-8")
    
    # Also save the combined one for legacy/comparison purposes
    combined_file = outdir / f"{pdf_stem}_{mode}_result.json"
    combined_file.write_text(eval_result_obj.model_dump_json(indent=2), encoding="utf-8")
    
    return combined_file

@app.command()
def evaluate(
    target: Path = typer.Argument(..., help="Path to the research paper PDF file or directory"),
    sample_size: Optional[int] = typer.Option(None, help="Number of papers to randomly sample"),
    subset: Optional[str] = typer.Option(None, help="Comma separated list of PDF stems to evaluate"),
    mode: str = typer.Option("fast", help="Execution mode ('fast', 'planning', 'both')"),
    profile: str = typer.Option("unpaid_lite", help="Configuration profile to use from eval_profiles.toml"),
    with_justification: bool = typer.Option(True, help="Whether to include justifications in the output"),
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
    
    cache_dir = Path("resources/markdown_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(run_outdir / "run.log")
    prompt_text = load_prompt(str(prompt))
    
    async def run_all():
        tasks = []
        for pdf in pdfs:
            if mode in ("fast", "both"):
                tasks.append(process_single_pdf(pdf, "fast", prompt_text, expected, run_outdir, with_justification, cache_dir))
            if mode in ("planning", "both"):
                tasks.append(process_single_pdf(pdf, "planning", prompt_text, expected, run_outdir, with_justification, cache_dir))
                
        results = await asyncio.gather(*tasks)
        
        if mode == "both":
            for i in range(0, len(results), 2):
                f_res = results[i]
                p_res = results[i+1]
                if f_res and p_res:
                    pdf_stem = pdfs[i//2].stem
                    report = compare_results(str(f_res), str(p_res))
                    highlight_print(f"--- Comparison Report for {pdf_stem} ---")
                    typer.echo(report)
                    (run_outdir / f"{pdf_stem}_comparison.md").write_text(report, encoding="utf-8")
                    
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
