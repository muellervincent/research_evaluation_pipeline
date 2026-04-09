import argparse
from pathlib import Path
from agent import process_pdf, run_fast_mode, run_planning_mode
from config import load_prompt
from evaluate import load_expected_answers, evaluate_fast_csv, evaluate_planning_json, compare_results
from datetime import datetime
import json
import logging
import random
from config import settings
from schema import EvaluationResult, EvaluationMetrics

def process_target(target_path: Path, sample_size: int = None) -> list[Path]:
    if target_path.is_file():
        return [target_path]
    elif target_path.is_dir():
        pdfs = list(target_path.glob("*.pdf"))
        # Ensure consistent order before sampling
        pdfs.sort()
        if sample_size and sample_size < len(pdfs):
            return random.sample(pdfs, sample_size)
        return pdfs
    return []

def main():
    parser = argparse.ArgumentParser(description="AI-Assisted RRP Analysis Tool")
    parser.add_argument("--target", type=str, help="Path to the research paper PDF file or directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of papers to randomly sample if target is a directory")
    parser.add_argument("--mode", choices=["fast", "planning"], default="fast", help="Execution mode")
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
    
    # Configure logging for this run (file=DEBUG/INFO, console=INFO)
    log_file = outdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting evaluation using model: {settings.model}")
    logging.info(f"Execution mode: {args.mode}")
    logging.info(f"Loading prompt from: {args.prompt}")
    prompt_text = load_prompt(args.prompt)
    
    for pdf_path in pdfs:
        pdf_stem = pdf_path.stem
        logging.info(f"\n--- Processing {pdf_stem} ---")
        try:
            markdown_text = process_pdf(str(pdf_path))
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            continue
            
        eval_result_obj = EvaluationResult(
            mode=args.mode,
            pdf_stem=pdf_stem,
            model_name=settings.model,
            metrics=None,
            raw_output=None
        )
        
        if args.mode == "fast":
            logging.info("Running Fast Mode...")
            csv_result = run_fast_mode(markdown_text, prompt_text)
            eval_result_obj.raw_output = csv_result
            
            try:
                expected = load_expected_answers(args.expected, pdf_stem)
                metrics_dict = evaluate_fast_csv(csv_result, expected)
                eval_result_obj.metrics = EvaluationMetrics(**metrics_dict)
                acc = metrics_dict.get('accuracy', 0)
                corr = metrics_dict.get('correct', 0)
                tot = metrics_dict.get('total', 0)
                logging.info(f"Accuracy: {acc*100:.2f}% ({corr}/{tot})")
            except Exception as e:
                logging.error(f"Could not perform evaluation: {e}")
                
        elif args.mode == "planning":
            logging.info("Running Planning Mode...")
            report_artifact = run_planning_mode(markdown_text, prompt_text)
            eval_result_obj.raw_output = report_artifact.model_dump()
            
            try:
                expected = load_expected_answers(args.expected, pdf_stem)
                metrics_dict = evaluate_planning_json(report_artifact.model_dump(), expected)
                eval_result_obj.metrics = EvaluationMetrics(**metrics_dict)
                acc = metrics_dict.get('accuracy', 0)
                corr = metrics_dict.get('correct', 0)
                tot = metrics_dict.get('total', 0)
                logging.info(f"Accuracy: {acc*100:.2f}% ({corr}/{tot})")
            except Exception as e:
                logging.error(f"Could not perform evaluation: {e}")
                
        # Save single unified result json
        out_file = outdir / f"{pdf_stem}_{args.mode}_result.json"
        out_file.write_text(eval_result_obj.model_dump_json(indent=2), encoding="utf-8")
        logging.info(f"Saved run result to {out_file}")

if __name__ == "__main__":
    main()
