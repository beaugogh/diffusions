from copy import deepcopy
from pathlib import Path
import yaml
import json
from typing import Mapping, Any


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def infer_model_traits(repo_id: str) -> dict:
    """
    Infer model traits from repo_id.
    Extend this as needed.
    """
    repo = repo_id.lower()

    traits = {
        "is_flux": "flux" in repo,
        "is_flux_schnell": "flux" in repo and "schnell" in repo,
    }

    return traits


def normalize_config(cfg: dict) -> dict:
    cfg = deepcopy(cfg)

    model = cfg["model"]
    gen = cfg["generation"]

    traits = infer_model_traits(model["repo_id"])

    # ---- FLUX-schnell invariants ----
    if traits["is_flux_schnell"]:
        # Hard invariant
        gen["guidance_scale"] = 0.0

        # Soft default (only if user didn't override)
        if gen.get("num_inference_steps") is None:
            gen["num_inference_steps"] = 20

        # Safety cap
        if gen.get("max_sequence_length", 512) > 256:
            gen["max_sequence_length"] = 256

    return cfg


def resolve_config(base_cfg: dict, override_cfg: dict | None = None) -> dict:
    merged = deep_merge(base_cfg, override_cfg or {})
    resolved = normalize_config(merged)
    return resolved


def load_config(
    config_path: str | Path,
    override_cfg: dict | None = None,
) -> dict:
    """
    Load YAML config from disk and resolve it with overrides.

    This is the ONLY place that touches YAML.
    """
    config_path = Path(config_path).expanduser().resolve()

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    resolved_config = resolve_config(base_cfg, override_cfg)
    print(json.dumps(resolved_config, ensure_ascii=False, indent=2))

    return resolved_config
