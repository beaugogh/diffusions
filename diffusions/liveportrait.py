# coding: utf-8
import os
import os.path as osp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
liveportrait_dir = str(os.path.join(current_dir, "..", "LivePortrait"))

import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


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


def main(driving=None, source=None, output_dir=None):
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

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

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    live_portrait_pipeline.execute(args)


if __name__ == "__main__":
    driving = osp.join(current_dir, "..", "assets", "liveportrait", "driving", "d3.mp4")
    source = osp.join(current_dir, "..", "assets", "liveportrait", "source", "s1.jpg")
    output_dir = osp.join(current_dir, "..", "assets", "liveportrait", "animations")
    main(driving, source, output_dir)
