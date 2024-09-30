# coding: utf-8
import os
import os.path as osp
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
liveportrait_dir = str(os.path.join(current_dir, "..", "LivePortrait"))

import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main(
    driving=None, source=None, output_dir=None, mode="human", driving_multiplier=1.0
):
    ###### default args ######
    # flag_use_half_precision: True
    # flag_crop_driving_video: False
    # device_id: 0
    # flag_force_cpu: False
    # flag_normalize_lip: False
    # flag_source_video_eye_retargeting: False
    # flag_eye_retargeting: False
    # flag_lip_retargeting: False
    # flag_stitching: True
    # flag_relative_motion: True
    # flag_pasteback: True
    # flag_do_crop: True
    # driving_option: expression-friendly
    # driving_multiplier: 1.0
    # driving_smooth_observation_variance: 3e-07
    # audio_priority: driving
    # animation_region: all
    # det_thresh: 0.15
    # scale: 2.3
    # vx_ratio: 0
    # vy_ratio: -0.125
    # flag_do_rot: True
    # source_max_dim: 1280
    # source_division: 2
    # scale_crop_driving_video: 2.2
    # vx_ratio_crop_driving_video: 0.0
    # vy_ratio_crop_driving_video: -0.1
    # server_port: 8890
    # share: False
    # server_name: 127.0.0.1
    # flag_do_torch_compile: False

    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    args.driving_multiplier = driving_multiplier
    # args.flag_stitching = False

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # customize conifg
    if driving is not None:
        args.driving = driving
    if source is not None:
        args.source = source
    if output_dir is not None:
        args.output_dir = output_dir

    # print("args: ", args)
    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    if mode == "human":
        pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    elif mode == "animal":
        pipeline = LivePortraitPipelineAnimal(
            inference_cfg=inference_cfg, crop_cfg=crop_cfg
        )
    else:
        raise Exception('unsupported mode, must be either "human" or "animal"!')

    # run
    pipeline.execute(args)


if __name__ == "__main__":
    driving = osp.join(
        current_dir, "..", "assets", "live_portrait", "driving", "driving_beau2.mp4"
    ) 
    source = osp.join(
        current_dir, "..", "assets", "live_portrait", "source", "han.jpg"
    )
    source = "/home/bo/workspace/diffusions/assets/live_portrait/source/杨乃文/ynw1-square-modified-out-squared.png"
    output_dir = osp.join(current_dir, "..", "assets", "live_portrait", "animations")
    mode = "human"  # human, animal
    driving_multiplier = 1.1
    main(
        driving=driving,
        source=source,
        output_dir=output_dir,
        mode=mode,
        driving_multiplier=driving_multiplier,
    )


