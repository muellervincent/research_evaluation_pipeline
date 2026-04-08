import argparse
from pathlib import Path
from agent import process_pdf, run_fast_mode, run_planning_mode
from config import load_prompt
from evaluate import load_expected_answers, evaluate_fast_csv, evaluate_planning_json
from datetime import datetime
import json
import logging
from config import settings

def main():
    parser = argparse.ArgumentParser(description="AI-Assisted RRP Analysis Tool")
    parser.add_argument("pdf_path", type=str, help="Path to the research paper PDF file")
    parser.add_argument("--mode", choices=["fast", "planning"], default="fast", help="Execution mode")
    parser.add_argument("--prompt", type=str, default="/Users/vincentmueller/Developer/Data/study_evalutation/resources/prompts.json", help="Path to the JSON file containing the master prompt")
    parser.add_argument("--outdir", type=str, default="./output", help="Directory to save the outputs")
    parser.add_argument("--expected", type=str, default="/Users/vincentmueller/Developer/Data/study_evalutation/resources/csv/correct_answers.CSV", help="Path to the ground truth CSV for evaluation")
    
    args = parser.parse_args()
    
    outdir_base = Path(args.outdir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = outdir_base / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(args.pdf_path).stem
    
    # Configure logging for this run
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
    
    logging.info(f"Converting PDF to Markdown: {args.pdf_path}")
    markdown_text = process_pdf(args.pdf_path)
    
    # Optionally save the markdown for auditability
    md_out = outdir / f"{pdf_stem}.md"
    md_out.write_text(markdown_text, encoding="utf-8")
    
    if args.mode == "fast":
        logging.info("Running in Fast Mode...")
        result = run_fast_mode(markdown_text, prompt_text)
        out_file = outdir / f"{pdf_stem}_fast.csv"
        out_file.write_text(result, encoding="utf-8")
        logging.info(f"Results saved to {out_file}")
        
        try:
            expected = load_expected_answers(args.expected, pdf_stem)
            eval_result = evaluate_fast_csv(out_file, expected)
            eval_out = outdir / f"{pdf_stem}_fast_eval.json"
            eval_out.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
            logging.info(f"Evaluation Accuracy: {eval_result['accuracy']*100:.2f}% ({eval_result['correct']}/{eval_result['total']})")
        except Exception as e:
            logging.error(f"Could not perform evaluation: {e}")
    
    elif args.mode == "planning":
        logging.info("Running in Planning Mode (this will take longer)...")
        report_artifact = run_planning_mode(markdown_text, prompt_text)
        out_file = outdir / f"{pdf_stem}_planning.json"
        out_file.write_text(report_artifact.model_dump_json(indent=2), encoding="utf-8")
        logging.info(f"Results saved to {out_file}")
        
        try:
            expected = load_expected_answers(args.expected, pdf_stem)
            eval_result = evaluate_planning_json(out_file, expected)
            eval_out = outdir / f"{pdf_stem}_planning_eval.json"
            eval_out.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
            logging.info(f"Evaluation Accuracy: {eval_result['accuracy']*100:.2f}% ({eval_result['correct']}/{eval_result['total']})")
        except Exception as e:
            logging.error(f"Could not perform evaluation: {e}")

if __name__ == "__main__":
    main()
