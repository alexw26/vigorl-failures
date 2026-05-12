import subprocess
import time
import requests
import os
import signal
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _vllm_headers(config: "BenchmarkConfig") -> dict[str, str]:
    if not config.api_key:
        return {}

    return {"Authorization": f"Bearer {config.api_key}"}

@dataclass
class BenchmarkConfig:
    # Pretrained model
    model_name: str
    
    # VLLM parameters
    n_gpus: int = 1
    port: int = 9001
    gpu_memory_utilization: float = 0.9
    startup_timeout_sec: int = 600
    request_timeout_sec: int = 2
    limit_mm_per_prompt: str = "image=30"
    mm_processor_kwargs: str = '{"max_pixels":12960000,"min_pixels":4096}'
    served_model_name: str = "qwen_vllm"
    api_key: str = "qwen"
    
    # ViGoRL parameters
    judge: str = "string_match"
    n_processes: int = 1
    n_rollouts: int = 1
    max_depth: int = 10
    max_new_tokens: int = 2048
    seed: int = 42
    checkpoint_interval: int = 10
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 10
    search_method: str = "dfs"
    repetition_penalty: float = 1.05
    
    
def data_dir() -> Path:
    return Path("/projects/bheg/sycui/grounded_rl/")


def _is_vllm_ready(config: BenchmarkConfig) -> bool:
    try:
        response = requests.get(
            f"http://localhost:{config.port}/v1/models",
            headers=_vllm_headers(config),
            timeout=config.request_timeout_sec,
        )
        if response.status_code != 200:
            logger.info(
                "vLLM readiness probe returned %s: %s",
                response.status_code,
                response.text[:200],
            )
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_vllm_server(config: BenchmarkConfig) -> subprocess.Popen | None:
    if _is_vllm_ready(config):
        logger.info(
            "vLLM server is already running on port %s. Reusing existing server.",
            config.port,
        )
        return None

    command = [
        "vllm",
        "serve",
        config.model_name,
        "--port",
        str(config.port),
        "--served-model-name",
        config.served_model_name,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(config.n_gpus),
        "--uvicorn-log-level",
        "info",
        "--limit-mm-per-prompt",
        config.limit_mm_per_prompt,
        "--mm-processor-kwargs",
        config.mm_processor_kwargs,
        "--api-key",
        config.api_key,
    ]
    logger.info("Launching vLLM command: %s", subprocess.list2cmdline(command))

    vllm_process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
        preexec_fn=os.setsid,
        env={**os.environ, "PORT": str(config.port)},
    )

    _POLL_SEC = 20
    logger.info("Waiting for vLLM server to become responsive...")
    deadline = time.time() + config.startup_timeout_sec
    while time.time() < deadline:
        if vllm_process.poll() is not None:
            raise RuntimeError("vLLM process exited before becoming ready.")

        if _is_vllm_ready(config):
            logger.info("vLLM server is up.")
            return vllm_process

        remaining = deadline - time.time()
        logger.warning(
            "vLLM not ready yet, retrying in %ss (%.0fs remaining)...",
            _POLL_SEC,
            remaining,
        )
        time.sleep(_POLL_SEC)

    os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
    msg = f"vLLM failed to start within {config.startup_timeout_sec} seconds."
    raise TimeoutError(msg)


def stop_vllm_server(vllm_process: subprocess.Popen | None) -> None:
    """Terminate a vLLM process started by start_vllm_server."""
    if vllm_process is None:
        return
    if vllm_process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
        vllm_process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)    
