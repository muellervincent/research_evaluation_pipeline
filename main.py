import argparse
from pathlib import Path
from agent import process_pdf, run_fast_mode, run_planning_mode
from config import load_prompt, settings
from evaluate import load_expected_answers, evaluate_report, compare_results
from datetime import datetime
import json
import random
from schema import EvaluationResult, EvaluationMetrics
from logger import setup_logging, highlight_print, console
import logging

def process_target(target_path: Path, sample_size: int = None) -> list[Path]:
    if target_path.is_file():
        return [target_path]
    elif target_path.is_dir():
        pdfs = list(target_path.glob("*.pdf"))
        pdfs.sort() # Ensure consistent order before sampling
        if sample_size and sample_size < len(pdfs):
            return random.sample(pdfs, sample_size)
        return pdfs
    return []

def run_evaluation(pdf_path: Path, mode: str, strategy: str, interactive: bool, prompt_text: str, expected_csv: str, outdir: Path) -> Path:
    pdf_stem = pdf_path.stem
    highlight_print(f"\n--- Processing {pdf_stem} [{mode.upper()} mode, Strategy: {strategy}] ---")
    
    try:
        markdown_text = process_pdf(str(pdf_path))
    except Exception as e:
        logging.error(f"Error extracting PDF {pdf_path}: {e}")
        return None
        
    eval_result_obj = EvaluationResult(
        mode=mode,
        pdf_stem=pdf_stem,
        model_name=settings.reasoning_model,
        metrics=None,
        raw_output=None
    )
    
    expected = load_expected_answers(expected_csv, pdf_stem)
    
    try:
        if mode == "fast":
            logging.info("Running Fast Mode...")
            result_artifact = run_fast_mode(markdown_text, prompt_text, interactive=interactive, strategy=strategy)
            eval_result_obj.raw_output = result_artifact.model_dump()
            metrics_dict = evaluate_report(result_artifact.model_dump(), expected)
            
        elif mode == "planning":
            logging.info("Running Planning Mode...")
            result_artifact = run_planning_mode(
                markdown_text, 
                prompt_text, 
                interactive, 
                strategy=strategy, 
                outdir=outdir
            )
            eval_result_obj.raw_output = result_artifact.model_dump()
            metrics_dict = evaluate_report(result_artifact.model_dump(), expected)
            
        # Add metrics
        eval_result_obj.metrics = EvaluationMetrics(**metrics_dict)
        acc = metrics_dict.get('accuracy', 0)
        corr = metrics_dict.get('correct', 0)
        tot = metrics_dict.get('total', 0)
        highlight_print(f"Outcome => Accuracy: {acc*100:.2f}% ({corr}/{tot})\n")
        
    except Exception as e:
        logging.error(f"Execution or evaluation failed: {e}")
        return None
        
    # Save unified result json
    out_file = outdir / f"{pdf_stem}_{mode}_result.json"
    out_file.write_text(eval_result_obj.model_dump_json(indent=2), encoding="utf-8")
    logging.info(f"Saved run result to {out_file}")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="AI-Assisted RRP Analysis Tool")
    parser.add_argument("--target", type=str, help="Path to the research paper PDF file or directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of papers to randomly sample if target is a directory")
    parser.add_argument("--mode", choices=["fast", "planning", "both"], default="fast", help="Execution mode")
    parser.add_argument("--strategy", choices=["batch", "isolated"], default="batch", help="Execution Strategy for breaking down tasks")
    parser.add_argument("--interactive", action="store_true", help="Pause for human-in-the-loop review of artifacts")
    parser.add_argument("--prompt", type=str, default="/Users/vincentmueller/Developer/Data/study_evalutation/resources/prompts.json", help="Path to the JSON file containing the master prompt")
    parser.add_argument("--outdir", type=str, default="./output", help="Directory to save the outputs")
    parser.add_argument("--expected", type=str, default="/Users/vincentmueller/Developer/Data/study_evalutation/resources/csv/correct_answers.CSV", help="Path to the ground truth CSV for evaluation")
    parser.add_argument("--compare", nargs=2, metavar=('FILE1', 'FILE2'), help="Compare two result JSON files and exit")
    
    args = parser.parse_args()
    
    if args.compare:
        print(f"Comparing {args.compare[0]} and {args.compare[1]}...\n")
        report = compare_results(args.compare[0], args.compare[1])
        print(report)
        return
        
    if not args.target:
        parser.error("the following arguments are required: --target (unless using --compare)")
        
    target_path = Path(args.target)
    pdfs = process_target(target_path, args.sample_size)
    if not pdfs:
        print("No PDFs found to process.")
        return
        
    outdir_base = Path(args.outdir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = outdir_base / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Configure custom logging (mutes noisy genai logs automatically)
    log_file = outdir / "run.log"
    logger = setup_logging(str(log_file))
    
    logger.info(f"Starting evaluation using Reasoning Model: {settings.reasoning_model} | Extraction: {settings.extraction_model}")
    logger.info(f"Execution mode: {args.mode.upper()}")
    logger.info(f"Strategy: {args.strategy.upper()}")
    prompt_text = load_prompt(args.prompt)
    
    for pdf_path in pdfs:
        # Run mode logic
        if args.mode == "both":
            fast_file = run_evaluation(pdf_path, "fast", args.strategy, args.interactive, prompt_text, args.expected, outdir)
            plan_file = run_evaluation(pdf_path, "planning", args.strategy, args.interactive, prompt_text, args.expected, outdir)
            if fast_file and plan_file:
                # Automate Comparison
                comparison_report = compare_results(str(fast_file), str(plan_file))
                highlight_print(f"--- Comparison Report for {pdf_path.stem} ---")
                console.print(comparison_report)
                (outdir / f"{pdf_path.stem}_comparison.md").write_text(comparison_report, encoding="utf-8")
        else:
            run_evaluation(pdf_path, args.mode, args.strategy, args.interactive, prompt_text, args.expected, outdir)

if __name__ == "__main__":
    main()
