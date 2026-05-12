import json
from io import BytesIO
import requests
import subprocess
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import datasets
from tqdm import tqdm

import helpers


def _is_valid_image_file(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return False


def _download_and_resize_image(image_url: str, image_path: Path) -> bool:
    """Generated via LLM."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.startswith("image/"):
            msg = f"URL {image_url!r} did not return an image content type, got {content_type!r}"
            raise ValueError(msg)

        with Image.open(BytesIO(response.content)) as image:
            image.load()
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(image_path)
        return True
    except (requests.RequestException, OSError, UnidentifiedImageError, ValueError) as _:
        image_path.unlink(missing_ok=True)
        return False


def convert_to_vlmsearch_jsonl(
    image_path: str | Path,
    jsonl_path: str | Path,
) -> None:
    data = datasets.load_dataset(
        path="johko/HaloQuest", 
        split="test", 
        cache_dir=helpers.data_dir()
    )
    
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    image_path.mkdir(parents=True, exist_ok=True)
    skipped_count = 0
        
    with jsonl_path.open("w", encoding="utf-8") as f:
        for id, data_dict in enumerate(tqdm(data)):
            img_file_name = image_path / data_dict["image_name"]
            if img_file_name.exists() and not _is_valid_image_file(img_file_name):
                img_file_name.unlink(missing_ok=True)

            if not img_file_name.exists():
                if not _download_and_resize_image(data_dict["url"], img_file_name):
                    skipped_count += 1
                    continue

            row = {
                "id": id,
                "image": str(img_file_name),
                "conversations": [
                    {"from": "human", "value": data_dict["question"]},
                    {"from": "gpt", "value": data_dict["groundtruth responses"]},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if skipped_count:
        print(f"Skipped {skipped_count} HaloQuest rows with unavailable or invalid images.")

def benchmark_haloquest(
    config: helpers.BenchmarkConfig,
    data_files: str | Path,
    image_root: str | Path,
) -> None:
    system_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. \n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, and reasoning steps must include specific reference coordinates:\n\nFor example, <think>\n{Reasoning text}. {Further reasoning text} {more reasoning} \n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> {Answer} </answer>\n"""
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
    # config = helpers.BenchmarkConfig(
    #     n_gpus=1,
    #     model_name="gsarch/ViGoRL-7b-Visual-Search",
    #     startup_timeout_sec=600,
    #     search_method="single_path_rollouts",
    #     checkpoint_interval=20
    # )
    
    convert_to_vlmsearch_jsonl(
        image_path=helpers.data_dir() / "haloquest/images",
        jsonl_path=helpers.data_dir() / "haloquest/validation.jsonl",
    )
    
    # vllm_process = helpers.start_vllm_server(config)

    # try:
    #     benchmark_haloquest(
    #         config,
    #         data_files=helpers.data_dir() / "haloquest/validation.jsonl",
    #         image_root=helpers.data_dir() / "haloquest/images",
    #     )
    # finally:
    #     helpers.stop_vllm_server(vllm_process)
    