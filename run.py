from src.pipeline import build_pipeline, text2image
from src.utils import save_image
from src.config import load_config


if __name__ == "__main__":
    # Runtime overrides (optional)
    override = {
        "model": {
            "repo_id": "/home/bo/workspace/models/black-forest-labs/FLUX.1-schnell"
        },
        "runtime": {"cpu_offload": True},
        "generation": {
            "prompt": "a tiny astronaut hatching from an egg on the moon",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 0,
            "max_sequence_length": 256,
        },
    }

    # 1. Load + resolve config
    cfg = load_config(
        config_path="config.yaml",
        override_cfg=override,
    )

    # 2. Build pipeline
    pipe = build_pipeline(cfg)

    # 3. Generate image
    image = text2image(
        pipe,
        cfg,
    )

    # 4. Save image
    output_path = save_image(
        image,
        output_dir="./outputs",
    )

    print(f"Image saved to: {output_path}")
