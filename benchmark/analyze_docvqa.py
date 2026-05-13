import json
from pathlib import Path

import pandas as pd


def parse_jsonl(file: str | Path) -> list[dict]:
    with open(file, "r") as f:
        return [json.loads(line) for line in f]
    
    
def docvqa_analyze():
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data/rollouts/gsarch_ViGoRL-7b-Visual-Search_docvqa_20260512_064731"
    data = list(data_dir.glob("rollouts*.jsonl")) 
    dicts = []
    for file in data:
        dicts.extend(parse_jsonl(file))
        
    count_failure = sum(1 for d in dicts if d["judge_score"] == 0)
    print(f"Failures: {count_failure} / {len(dicts)}, {count_failure / len(dicts) * 100:.2f}%")
    
    failures = []
    for d in dicts:
        if d["judge_score"] == 0:
            failures.append({
                "question": d["question"],
                "image": d["image"],
                "answer": d["final_answer"],
                "ground_truth": d["true_answer"],
                "thoughts": d["thoughts"]
            })
    
    # Save failures to file
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(project_dir / "docvqa_failures.csv", index=False)
        

if __name__ == "__main__":
    docvqa_analyze()
    # import helpers
    # data = helpers.data_dir() / "docvqa/validation.jsonl"
    # count_lines = sum(1 for _ in open(data, "r"))
    # print(count_lines)
    
