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

def evaluate_fast_csv(csv_path: str, expected: Dict[str, str]) -> Dict[str, Any]:
    correct = 0
    total = 0
    details = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.strip():
        return {"accuracy": 0, "correct": 0, "total": 0, "details": []}
        
    try:
        delim = csv.Sniffer().sniff(content[:1024]).delimiter
    except csv.Error:
        delim = ';' if ';' in content[:256] else ','
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delim)
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
                        "raw_output": ans_str,
                        "correct": is_correct
                    })
    
    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details
    }

def evaluate_planning_json(json_path: str, expected: Dict[str, str]) -> Dict[str, Any]:
    correct = 0
    total = 0
    details = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
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

