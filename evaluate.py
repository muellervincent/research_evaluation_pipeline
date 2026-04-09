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
    with open(expected_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        # Handle zero padding
        target_study_num = str(study_number).lstrip('0')
        if not target_study_num:
            target_study_num = '0'
            
        for row in reader:
            if str(row['study_number']).strip() == target_study_num:
                expected[str(row['prompt_number']).strip()] = str(row['answer']).strip()
    return expected

def evaluate_fast_csv(csv_content: str, expected: Dict[str, str]) -> Dict[str, Any]:
    correct = 0
    total = 0
    details = []
    
    if not csv_content.strip():
        return {"accuracy": 0, "correct": 0, "total": 0, "details": []}
        
    try:
        delim = csv.Sniffer().sniff(csv_content[:1024]).delimiter
    except csv.Error:
        delim = ';' if ';' in csv_content[:256] else ','
        
    reader = csv.reader(csv_content.splitlines(), delimiter=delim)
    for row in reader:
        if len(row) >= 2:
                q_num = str(row[0]).strip().rstrip('.')
                ans_str = row[1]
                ans = normalize_answer(ans_str)
                exp = expected.get(q_num)
                
                if exp is not None and exp != 'NA':
                    total += 1
                    is_correct = (ans == exp)
                    if is_correct:
                        correct += 1
                    details.append({
                        "question": q_num,
                        "expected": exp,
                        "actual": ans,
                        "correct": is_correct
                    })
    
    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details
    }

def evaluate_planning_json(data: Dict[str, Any], expected: Dict[str, str]) -> Dict[str, Any]:
    correct = 0
    total = 0
    details = []
        
    for answer_obj in data.get('answers', []):
        q_num = str(answer_obj.get('question_number')).strip()
        ans_bool = answer_obj.get('answer')
        ans = '1' if ans_bool else '0'
        exp = expected.get(q_num)
        
        if exp is not None and exp != 'NA':
            total += 1
            is_correct = (ans == exp)
            if is_correct:
                correct += 1
            details.append({
                "question": q_num,
                "expected": exp,
                "actual": ans,
                "correct": is_correct
            })
            
    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details
    }

def compare_results(file1: str, file2: str) -> str:
    """Compares two JSON result files and returns a markdown formatted diff."""
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        d1 = json.load(f1)
        d2 = json.load(f2)
    
    m1 = d1.get("metrics", {})
    if not m1: m1 = {}
    m2 = d2.get("metrics", {})
    if not m2: m2 = {}
    
    report = [f"## Comparison: {Path(file1).name} vs {Path(file2).name}"]
    report.append("| Metric | File 1 | File 2 | Diff |")
    report.append("|---|---|---|---|")
    
    acc1 = m1.get("accuracy", 0)
    acc2 = m2.get("accuracy", 0)
    report.append(f"| Accuracy | {acc1*100:.2f}% | {acc2*100:.2f}% | {(acc2-acc1)*100:+.2f}% |")
    
    corr1 = m1.get('correct', 0)
    corr2 = m2.get('correct', 0)
    report.append(f"| Correct | {corr1} | {corr2} | {corr2 - corr1} |")
    
    # Detail diff
    det1 = { d["question"]: d for d in m1.get("details", []) }
    det2 = { d["question"]: d for d in m2.get("details", []) }
    
    all_qs = sorted(set(det1.keys()).union(det2.keys()))
    diffs = []
    
    for q in all_qs:
        ans1 = det1.get(q, {}).get("actual", "N/A")
        ans2 = det2.get(q, {}).get("actual", "N/A")
        if ans1 != ans2:
            diffs.append(f"- **Q{q}**: File1='{ans1}' vs File2='{ans2}'")
            
    if diffs:
        report.append("\n### Diverging Answers\n")
        report.extend(diffs)
    else:
        report.append("\n### No Diverging Answers\n")
        
    return "\n".join(report)

