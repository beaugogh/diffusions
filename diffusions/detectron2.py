import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
detectron2_dir = str(os.path.join(current_dir, "..", "detectron2"))
sys.path.insert(0, detectron2_dir)

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, mplc
from detectron2.data import MetadataCatalog


class CustomVisualizer(Visualizer):
    def draw_instance_predictions_keypoints(self, predictions, jittering: bool = True):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
            jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
                to distinguish instances from the same class

        Returns:
            output (VisImage): image object with visualizations.
        """
        # self.img.fill(255)
        boxes = None
        scores = None
        classes = None
        labels = None
        keypoints = (
            predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        )

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [
                GenericMask(x, self.output.height, self.output.width) for x in masks
            ]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = (
                [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                    for c in classes
                ]
                if jittering
                else [
                    tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
                    for c in classes
                ]
            )

            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


class DetectronModel:
    def __init__(self):
        self.cfg = ""
        self.predictor = ""

    def predict(self):
        self.predictor()


def showimg(img, title="image"):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_model(yaml_path):
    print("loading...")
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(yaml_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_path)
    predictor = DefaultPredictor(cfg)
    print(f"model {yaml_path} loaded")
    return cfg, model


def predict_and_save(cfg, model, input_path, output_path):
    im = cv2.imread(input_path)
    print("predicting...")
    outputs = model(im)
    print("visualizing...")
    instances = outputs["instances"].to("cpu")
    # print(instances.pred_keypoints)
    # print(instances.pred_keypoint_heatmaps)
    im_white = im.copy()
    im_white.fill(255)
    v = CustomVisualizer(
        im_white, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    # # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = v.draw_instance_predictions_keypoints(instances)
    res = out.get_image()
    print("saving to ", output_path)
    cv2.imwrite(output_path, res)


if __name__ == "__main__":
    input_path = "/home/bo/workspace/artdiffusion/54 JuJutsu Techniques _ Self Defence Syllabus _ Traditional Japanese Ju Jutsu Ryu_frames/frame_158.jpg"
    output_path = "./images/keypoints2.png"
    yaml_path = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    # yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    cfg, model = load_model(yaml_path)
    predict_and_save(cfg, model, input_path, output_path)
