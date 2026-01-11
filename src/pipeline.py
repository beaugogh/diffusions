import torch
from diffusers import FluxPipeline
from .utils import is_notebook


"""this one is the first to work, in terms of loading the model, without blowing up the memory and crashing wsl"""
# def build_pipeline(cfg: dict) -> FluxPipeline:
#     model_cfg = cfg["model"]
#     runtime_cfg = cfg["runtime"]

#     repo_id: str = model_cfg["repo_id"]
#     dtype: torch.dtype = getattr(torch, runtime_cfg["dtype"])

#     pipe = FluxPipeline.from_pretrained(
#         repo_id,
#         torch_dtype=dtype,
#     )

#     # Device / offload handling
#     if runtime_cfg.get("cpu_offload", False):
#         pipe.enable_model_cpu_offload()
#     else:
#         pipe.to(runtime_cfg.get("device", "cuda"))

#     return pipe

"""this is the second iteration that works, trying to re-introduce some of the wsl-safe optimizations from pipeline.py. cpu_offload is true by default here."""
def build_pipeline(cfg: dict) -> FluxPipeline:
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    repo_id = model_cfg["repo_id"]
    dtype = getattr(torch, runtime_cfg.get("dtype", "float16"))
    device = runtime_cfg.get("device", "cuda")
    cpu_offload = runtime_cfg.get("cpu_offload", False)

    pipe = FluxPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,  # somehow using dtype does not work here, FluxPipeline does not accept it, and crashes wsl due to some memory issue
        local_files_only=True,
    )

    # Device / offload handling
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device, dtype=dtype)

    print(f"Pipeline ready | device={device}, dtype={dtype}, cpu_offload={cpu_offload}")

    return pipe

"""this is the third iteration, NOT working, takes forever to do steps, re-introducing torch.compile when cpu_offload is false and not in a notebook environment."""
# def build_pipeline(cfg: dict) -> FluxPipeline:
#     model_cfg = cfg["model"]
#     runtime_cfg = cfg["runtime"]

#     repo_id = model_cfg["repo_id"]
#     dtype = getattr(torch, runtime_cfg.get("dtype", "float16"))
#     device = runtime_cfg.get("device", "cuda")
#     cpu_offload = runtime_cfg.get("cpu_offload", False)

#     pipe = FluxPipeline.from_pretrained(
#         repo_id,
#         torch_dtype=dtype,  # somehow using dtype does not work here, FluxPipeline does not accept it, and crashes wsl due to some memory issue
#         local_files_only=True,
#     )

#     if not is_notebook() and not cpu_offload:
#         pipe.transformer = torch.compile(
#             pipe.transformer,
#             mode="reduce-overhead",
#             fullgraph=True,
#         )
#         print("torch.compile enabled for transformer.")
#     else:
#         print("torch.compile disabled")
#         print(f"is_notebook={is_notebook()}, cpu_offload={cpu_offload}")

#     # Device / offload handling
#     if cpu_offload:
#         pipe.enable_model_cpu_offload()
#     else:
#         pipe.to(device, dtype=dtype)

#     # ---- WSL-safe memory optimizations ----
#     # pipe.enable_attention_slicing("auto")
#     # pipe.vae.enable_tiling()
#     # pipe.set_progress_bar_config(disable=True)

#     # ---- TODO: torch.compile ----

#     print(f"Pipeline ready | device={device}, dtype={dtype}, cpu_offload={cpu_offload}")

#     return pipe


"""this is the fourth iteration, re-introducing wsl-safe optimizations, but keeping torch.compile disabled always. cpu_offload=False NOT working, too slow. cpu_offload=True working.
adding the wsl-safe optimizations seems to help a bit with speed when cpu_offload is true. latency from 01:25 to 00:48"""
def build_pipeline(cfg: dict) -> FluxPipeline:
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    repo_id = model_cfg["repo_id"]
    dtype = getattr(torch, runtime_cfg.get("dtype", "float16"))
    device = runtime_cfg.get("device", "cuda")
    cpu_offload = runtime_cfg.get("cpu_offload", False)

    pipe = FluxPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,  # somehow using dtype does not work here, FluxPipeline does not accept it, and crashes wsl due to some memory issue
        local_files_only=True,
    )

    # Device / offload handling
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device, dtype=dtype)

    # ---- WSL-safe memory optimizations ----
    pipe.enable_attention_slicing("auto")
    pipe.vae.enable_tiling()
    # this will disable the steps visual
    # pipe.set_progress_bar_config(disable=True)

    print(f"Pipeline ready | device={device}, dtype={dtype}, cpu_offload={cpu_offload}")

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
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        result = pipe(**kwargs)
        return result.images[0]
