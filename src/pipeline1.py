import torch
from diffusers import FluxPipeline
from typing import Any
from src.config import load_config
from src.utils import save_image, is_notebook


def build_pipeline(cfg: dict) -> FluxPipeline:
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    repo_id = model_cfg["repo_id"]
    dtype = getattr(torch, runtime_cfg.get("dtype", "float16"))
    device = runtime_cfg.get("device", "cuda")

    # ---- Hard guard: CPU offload disabled under WSL ----
    if runtime_cfg.get("cpu_offload", False):
        raise RuntimeError(
            "CPU offload is disabled. It is unsafe for large FLUX models under WSL2."
        )

    # ---- Load pipeline (CPU by default) ----
    pipe = FluxPipeline.from_pretrained(
        repo_id,
        local_files_only=True,
    )

    # ---- Move entire model to GPU ----
    pipe = pipe.to(device=device, dtype=dtype)

    # ---- WSL-safe memory optimizations ----
    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)

    # ---- Explicitly disable torch.compile ----
    print("torch.compile disabled (WSL-safe execution mode).")

    # ---- CUDA math tuning (Ampere-safe) ----
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Pipeline ready | device={device}, dtype={dtype}, cpu_offload=False")

    return pipe


def text2image(
    pipe,
    cfg: dict,
    *,
    prompt: str | None = None,
):
    gen_cfg = cfg["generation"]

    final_prompt = prompt or gen_cfg.get("prompt")
    if not final_prompt:
        raise ValueError("Prompt must be provided.")

    generator = None
    if gen_cfg.get("seed") is not None:
        generator = torch.Generator(pipe.device).manual_seed(gen_cfg["seed"])

    kwargs = {
        "prompt": final_prompt,
        "width": gen_cfg.get("width"),
        "height": gen_cfg.get("height"),
        "num_inference_steps": gen_cfg.get("num_inference_steps"),
        "guidance_scale": gen_cfg.get("guidance_scale"),
        "generator": generator,
        "output_type": "pil",
    }

    if gen_cfg.get("max_sequence_length") is not None:
        kwargs["max_sequence_length"] = gen_cfg["max_sequence_length"]

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # CUDA math tuning
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        result = pipe(**kwargs)
        return result.images[0]
