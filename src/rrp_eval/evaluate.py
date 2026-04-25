import csv
import json
from pathlib import Path
from typing import Dict, Any

def normalize_answer(ans_str: str) -> str:
    ans = str(ans_str).strip().lower()
    if ans in ['1', 'true', 'yes', 'y']:
        return '1'
    elif ans in ['0', 'false', 'no', 'n']:
        return '0'
    else:
        return 'NA'

def load_expected_answers(expected_csv_path: str, study_number: str) -> Dict[str, str]:
    expected = {}
    path = Path(expected_csv_path)
    if not path.exists():
        return expected
        
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        # Handle zero padding
        target_study_num = str(study_number).lstrip('0')
        if not target_study_num:
            target_study_num = '0'
            
        for row in reader:
            if str(row.get('study_number', '')).strip() == target_study_num:
                expected[str(row.get('prompt_number', '')).strip()] = str(row.get('answer', '')).strip()
    return expected

def evaluate_report(data: Dict[str, Any], expected: Dict[str, str]) -> Dict[str, Any]:
    correct = 0
    total = 0
    details = []
    
    # Create a mapping from question_number to justification from raw output
    justifications = {
        str(ans_obj.get('question_number')).strip().rstrip('.'): ans_obj.get('justification')
        for ans_obj in data.get('answers', [])
    }
        
    for answer_obj in data.get('answers', []):
        q_num = str(answer_obj.get('question_number')).strip().rstrip('.')
        ans_bool = answer_obj.get('answer')
        
        if isinstance(ans_bool, bool):
            ans = '1' if ans_bool else '0'
        else:
            ans = normalize_answer(ans_bool)

        exp = expected.get(q_num)
        
        if exp is not None and exp != 'NA':
            total += 1
            is_correct = (ans == exp)
            if is_correct:
                correct += 1
            details.append({
                "question_number": q_num,
                "expected": exp,
                "predicted": ans,
                "correct": is_correct,
                "justification": justifications.get(q_num)
            })
            
    missing_answers = max(0, len(expected) - len(data.get('answers', [])))
    
    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details,
        "missing_answers": missing_answers
    }

def compare_results(file1: str, file2: str) -> str:
    """Compares two JSON result files and returns a markdown formatted diff."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        d1 = json.load(f1)
        d2 = json.load(f2)
    
    m1 = d1.get("metrics", {}) or {}
    m2 = d2.get("metrics", {}) or {}
    
    # 1. Markdown Report
    report = [f"## Comparison: {Path(file1).name} vs {Path(file2).name}"]
    report.append("| Metric | File 1 | File 2 | Diff |")
    report.append("|---|---|---|---|")
    
    acc1 = m1.get("accuracy", 0)
    acc2 = m2.get("accuracy", 0)
    report.append(f"| Accuracy | {acc1*100:.2f}% | {acc2*100:.2f}% | {(acc2-acc1)*100:+.2f}% |")
    
    corr1 = m1.get('correct', 0)
    corr2 = m2.get('correct', 0)
    report.append(f"| Correct | {corr1} | {corr2} | {corr2 - corr1} |")
    
    det1 = { d["question_number"]: d for d in m1.get("details", []) }
    det2 = { d["question_number"]: d for d in m2.get("details", []) }
    
    all_qs = sorted(set(det1.keys()).union(det2.keys()))
    
    # 2. Console Visualization (Rich)
    table = Table(title=f"Comparison: {Path(file1).name} vs {Path(file2).name}")
    table.add_column("Question", style="cyan")
    table.add_column(f"File 1 ({d1.get('mode', 'fast')})", style="magenta")
    table.add_column(f"File 2 ({d2.get('mode', 'planning')})", style="green")
    table.add_column("Match", justify="center")

    diffs = []
    for q in all_qs:
        ans1 = det1.get(q, {}).get("predicted", "N/A")
        ans2 = det2.get(q, {}).get("predicted", "N/A")
        
        match_symbol = "[bold green]✓[/]" if ans1 == ans2 else "[bold red]✗[/]"
        table.add_row(f"Q{q}", str(ans1), str(ans2), match_symbol)
        
        if ans1 != ans2:
            diffs.append(f"- **Q{q}**: File1='{ans1}' vs File2='{ans2}'")
            
    console.print(table)

    if diffs:
        report.append("\n### Diverging Answers\n")
        report.extend(diffs)
    else:
        report.append("\n### No Diverging Answers\n")
        
    return "\n".join(report)
