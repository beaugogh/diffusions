import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
detectron2_dir = str(os.path.join(current_dir, "..", "detectron2"))
densepose_dir = str(
    os.path.join(current_dir, "..", "detectron2", "projects", "DensePose")
)
sys.path.insert(0, detectron2_dir)
sys.path.insert(0, densepose_dir)

import torch
import cv2
import numpy as np
from typing import List
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsVertexVisualizer,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    create_extractor,
)


class DensePoseModel:
    VISUALIZERS = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer, # not working atm
        "bbox": ScoredBoundingBoxVisualizer,
    }

    def __init__(self, config_path: str, model_path: str):
        print(f"load densepose model from {model_path}...")
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.freeze()
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        print("finished loading")

    def _visualize_outputs(
        self,
        input_img,
        outputs,
        vis_specs: List[str],
        save_path: str,
        white_background: bool = False,
    ):
        # create visualizer and extractor
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            vis = DensePoseModel.VISUALIZERS[vis_spec](
                cfg=self.cfg,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)

        # visualize and save
        image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        if white_background:
            image.fill(255)
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        cv2.imwrite(save_path, image_vis)

    def predict_and_save(
        self,
        input_img_path: str,
        output_img_path: str = None,
        vis_specs: str = "dp_contour,bbox",
        white_background: bool = False,
    ):
        if not output_img_path:
            suffix = input_img_path.split(".")[-1]
            output_img_path = input_img_path.replace(
                f".{suffix}", f"-{vis_specs}.{suffix}"
            )
        vis_specs = vis_specs.split(",")
        img = read_image(input_img_path, format="BGR")  # predictor expects BGR image.
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            self._visualize_outputs(
                img,
                outputs,
                vis_specs,
                output_img_path,
                white_background=white_background,
            )
            print(f"output is saved to {output_img_path}")


if __name__ == "__main__":
    # example command: python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml models/densepose_rcnn_R_50_FPN_s1x.pkl images/frame_148.jpg dp_segm --output images/image_densepose_contour_output.png
    model_path = "/home/bo/workspace/diffusions/models/densepose_rcnn_R_50_FPN_s1x.pkl"
    config_path = "/home/bo/workspace/diffusions/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
    input_img_path = "/home/bo/workspace/diffusions/images/frame_156.jpg"
    # available specs: dp_contour,dp_segm,dp_u,dp_v,dp_vertex,bbox
    specs = "dp_segm,bbox" 
    white_bkg = True

    model = DensePoseModel(config_path, model_path)
    model.predict_and_save(input_img_path, vis_specs=specs, white_background=white_bkg)
