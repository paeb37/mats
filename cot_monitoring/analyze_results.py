from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re

import fire


def _load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def analyze_graded_results(graded_eval_jsonl: str, output_json: str | None = None) -> dict:
    rows = _load_jsonl(graded_eval_jsonl)
    if not rows:
        raise ValueError("No rows found")
        
    total = len(rows)
    
    # 1. Aggregate Stats
    correct_count = sum(1 for r in rows if r.get("grade", {}).get("is_correct"))
    unfaithful_count = sum(1 for r in rows if r.get("grade", {}).get("is_unfaithful"))
    
    accuracy = correct_count / total if total > 0 else 0
    unfaithful_rate = unfaithful_count / total if total > 0 else 0
    
    # 2. Pair Analysis (Check for Evidence Sensitivity)
    # Pairs share the same ID prefix: "case_{XXX}_{variant}"
    # e.g., "case_000_a", "case_000_b"
    pairs = {}
    for r in rows:
        case_id = r.get("case", {}).get("id")
        match = re.match(r"(case_\d+)_([ab])", case_id)
        if match:
            base_id = match.group(1)
            variant = match.group(2)
            if base_id not in pairs:
                pairs[base_id] = {}
            pairs[base_id][variant] = r
            
    consistent_flips = 0
    total_complete_pairs = 0
    
    for base_id, variants in pairs.items():
        if "a" in variants and "b" in variants:
            total_complete_pairs += 1
            # Check if both A and B were answered correctly
            is_correct_a = variants["a"].get("grade", {}).get("is_correct")
            is_correct_b = variants["b"].get("grade", {}).get("is_correct")
            
            if is_correct_a and is_correct_b:
                consistent_flips += 1
                
    flip_consistency = consistent_flips / total_complete_pairs if total_complete_pairs > 0 else 0

    metrics = {
        "n_samples": total,
        "n_pairs": total_complete_pairs,
        "accuracy": accuracy,
        "unfaithful_rate": unfaithful_rate,
        "evidence_sensitivity_rate": flip_consistency,  # % of pairs where model correctly followed the toggle both times
    }

    out = {
        "graded_eval_jsonl": str(graded_eval_jsonl),
        "metrics": metrics,
    }
    
    print("\n=== ANALYSIS RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("========================\n")
    
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote analysis to {out_path}")
        
    return out


if __name__ == "__main__":
    fire.Fire(analyze_graded_results)