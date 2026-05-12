import hashlib
import json
import subprocess
from pathlib import Path
from PIL import Image

import datasets
from tqdm import tqdm

import helpers


def convert_to_vlmsearch_jsonl(
    image_path: str | Path,
    jsonl_path: str | Path,
) -> None:
    data = datasets.load_dataset(
        path="lmms-lab/DocVQA", 
        name="DocVQA", 
        split="validation", 
        cache_dir=helpers.data_dir()
    )
    
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    image_path.mkdir(parents=True, exist_ok=True)
    
    image_set = {}
        
    with jsonl_path.open("w", encoding="utf-8") as f:
        for data_dict in tqdm(data):
            img_file_name = image_path / f"{data_dict['questionId']}.jpeg"
            image = data_dict["image"]
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            if image_hash not in image_set:
                image_set[image_hash] = img_file_name
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                image.save(img_file_name)

            row = {
                "id": data_dict["questionId"],
                "image": str(image_set[image_hash]),
                "conversations": [
                    {"from": "human", "value": data_dict["question"]},
                    {"from": "gpt", "value": data_dict["answers"]},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def benchmark_docvqa(
    config: helpers.BenchmarkConfig,
    data_files: str | Path,
    image_root: str | Path,
) -> None:
    system_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. There may be one image or two images concatenated together, in which case the Assistant must compare the spatial relationships between the two images.\n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, and reasoning steps must include specific reference coordinates:\n\nFor example, <think>\n{Reasoning text}. {Further reasoning text} {more reasoning} \n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> {text of selected answer choice} </answer>\n\nThe Assistant must help the user identify the correct answer choice from the options provided.\n-Your answer should be the **exact text** of the selected answer option, without additional explanations or reasoning or the option text. For example, if the answer is A. right , your response should just be <answer>right</answer> (not <answer>A. right</answer>).\n-If the correct answer is unclear, select the most relevant option based on the spatial relationships and dynamics within the image.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."""
    cmds = [
        "python",
        "-m", "src.vlmsearch",
        "--seed", str(config.seed),
        "--model", config.served_model_name,
        "--judge", config.judge,
        "--search_method", config.search_method,
        "--max_depth", str(config.max_depth),
        "--n_rollouts", str(config.n_rollouts),
        "--data_files", str(data_files),
        "--image_root", str(image_root),
        "--temperature", str(config.temperature),
        "--top_p", str(config.top_p),
        "--top_k", str(config.top_k),
        "--max_new_tokens", str(config.max_new_tokens),
        "--num_processes", str(config.n_processes),
        "--do_data_checkpoint",
        "--checkpoint_interval", str(config.checkpoint_interval),
        "--pretrained", config.model_name,
        "--max_samples", "5349",
        "--system_prompt", system_prompt,
        "--generate_upfront",
        # "--first_rollout_no_sample",
        "--save_tag", f"{config.model_name}_docvqa",
        "--repetition_penalty", str(config.repetition_penalty),
    ]
    subprocess.run(cmds, check=True)
    
    
if __name__ == "__main__":
    config = helpers.BenchmarkConfig(
        n_gpus=4,
        model_name="gsarch/ViGoRL-7b-Visual-Search",
        startup_timeout_sec=1200,
        search_method="single_path_rollouts",
        checkpoint_interval=200
    )
    
    # convert_to_vlmsearch_jsonl(
    #     image_path=helpers.data_dir() / "docvqa/images",
    #     jsonl_path=helpers.data_dir() / "docvqa/validation.jsonl",
    # )
    
    vllm_process = helpers.start_vllm_server(config)

    try:
        benchmark_docvqa(
            config,
            data_files=helpers.data_dir() / "docvqa/validation.jsonl",
            image_root=helpers.data_dir() / "docvqa/images",
        )
    finally:
        helpers.stop_vllm_server(vllm_process)
    