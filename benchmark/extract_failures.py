import json
from pathlib import Path

import click
import pandas as pd


def parse_jsonl(file: str | Path) -> list[dict]:
    with open(file, "r") as f:
        return [json.loads(line) for line in f]
    

@click.command()
@click.option("--input_dir", type=click.Path(exists=True))
@click.option("--output_file", type=click.Path())
def extract_failures(
    input_dir: str | Path, output_file: str | Path
) -> None:
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    data = list(input_dir.glob("rollouts*.jsonl")) 
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
    failures_df.to_csv(output_file, index=False)
        

if __name__ == "__main__":
    extract_failures()
    
