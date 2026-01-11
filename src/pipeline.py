import torch
from diffusers import FluxPipeline
from typing import Any
from src.config import load_config
from src.utils import save_image, is_notebook



def build_pipeline(cfg: dict) -> FluxPipeline:
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    repo_id = model_cfg["repo_id"]
    dtype = getattr(torch, runtime_cfg["dtype"])
    device = runtime_cfg.get("device", "cuda")

    pipe = FluxPipeline.from_pretrained(
        repo_id,
        local_files_only=True,
    )

    if not is_notebook():
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=True,
        )
        print("Not in notebook environment, apply pipeline compiling...")
    else:
        print("In notebook environment.")

    if runtime_cfg.get("cpu_offload", False):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device=device, dtype=dtype)

    return pipe


def text2image(
    pipe,
    cfg: dict,
    *,
    prompt: str | None = None,
) -> Any:
    """
    Generic text-to-image generation.

    Assumes:
    - cfg has already been resolved & normalized
    - pipe has already been constructed and placed on device

    Returns:
        The first generated PIL image (Diffusers standard behavior)
    """
    gen_cfg = cfg["generation"]

    # Prompt resolution order:
    # 1. Explicit function argument
    # 2. Config value
    # 3. Error
    final_prompt = prompt or gen_cfg.get("prompt")
    if not final_prompt:
        raise ValueError("Prompt must be provided either as argument or in config.")

    # Optional deterministic seeding
    generator = None
    if gen_cfg.get("seed") is not None:
        generator = torch.Generator("cpu").manual_seed(gen_cfg["seed"])

    # Build kwargs dynamically (keeps this generic)
    kwargs = {
        "prompt": final_prompt,
        "width": gen_cfg.get("width"),
        "height": gen_cfg.get("height"),
        "num_inference_steps": gen_cfg.get("num_inference_steps"),
        "guidance_scale": gen_cfg.get("guidance_scale"),
        "generator": generator,
    }

    # Some pipelines (e.g. FLUX) support max_sequence_length
    if gen_cfg.get("max_sequence_length") is not None:
        kwargs["max_sequence_length"] = gen_cfg["max_sequence_length"]

    # Remove None values (important for pipeline compatibility)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    with torch.inference_mode():
        result = pipe(**kwargs)
        return result.images[0]

