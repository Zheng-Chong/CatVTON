
import glob
import os
from random import randint
import shutil
import time

import cv2
import numpy as np
import torch
from PIL import Image
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import create_extractor, CompoundExtractor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor


class DensePose:
    """
    DensePose used in this project is from Detectron2 (https://github.com/facebookresearch/detectron2).
    These codes are modified from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose.
    The checkpoint is downloaded from https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo.

    We use the model R_50_FPN_s1x with id 165712039, but other models should also work.
    The config file is downloaded from https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs.
    Noted that the config file should match the model checkpoint and Base-DensePose-RCNN-FPN.yaml is also needed.
    """

    def __init__(self, model_path="./checkpoints/densepose_", device="cuda"):
        self.device = device
        self.config_path = os.path.join(model_path, 'densepose_rcnn_R_50_FPN_s1x.yaml')
        self.model_path = os.path.join(model_path, 'model_final_162be9.pkl')
        self.visualizations = ["dp_segm"]
        self.VISUALIZERS = {"dp_segm": DensePoseResultsFineSegmentationVisualizer}
        self.min_score = 0.8

        self.cfg = self.setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.to(self.device)

    def setup_config(self):
        opts = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(self.min_score)]
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.freeze()
        return cfg

    @staticmethod
    def _get_input_file_list(input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [os.path.join(input_spec, fname) for fname in os.listdir(input_spec)
                         if os.path.isfile(os.path.join(input_spec, fname))]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    def create_context(self, cfg, output_path):
        vis_specs = self.visualizations
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = texture_atlases_dict = None
            vis = self.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
                alpha=1.0
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": output_path,
            "entry_idx": 0,
        }
        return context

    def execute_on_outputs(self, context, entry, outputs):
        extractor = context["extractor"]

        data = extractor(outputs)

        H, W, _ = entry["image"].shape
        result = np.zeros((H, W), dtype=np.uint8)

        data, box = data[0]
        x, y, w, h = [int(_) for _ in box[0].cpu().numpy()]
        i_array = data[0].labels[None].cpu().numpy()[0]
        result[y:y + h, x:x + w] = i_array
        result = Image.fromarray(result)
        result.save(context["out_fname"])

    def __call__(self, image_or_path, resize=512) -> Image.Image:
        """
        :param image_or_path: Path of the input image.
        :param resize: Resize the input image if its max size is larger than this value.
        :return: Dense pose image.
        """
        # random tmp path with timestamp
        tmp_path = f"./densepose_/tmp/"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        image_path = os.path.join(tmp_path, f"{int(time.time())}-{self.device}-{randint(0, 100000)}.png")
        if isinstance(image_or_path, str):
            assert image_or_path.split(".")[-1] in ["jpg", "png"], "Only support jpg and png images."
            shutil.copy(image_or_path, image_path)
        elif isinstance(image_or_path, Image.Image):
            image_or_path.save(image_path)
        else:
            shutil.rmtree(tmp_path)
            raise TypeError("image_path must be str or PIL.Image.Image")

        output_path = image_path.replace(".png", "_dense.png").replace(".jpg", "_dense.png")
        w, h = Image.open(image_path).size

        file_list = self._get_input_file_list(image_path)
        assert len(file_list), "No input images found!"
        context = self.create_context(self.cfg, output_path)
        for file_name in file_list:
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            # resize
            if (_ := max(img.shape)) > resize:
                scale = resize / _
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

            with torch.no_grad():
                outputs = self.predictor(img)["instances"]
                try:
                    self.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
                except Exception as e:
                    null_gray = Image.new('L', (1, 1))
                    null_gray.save(output_path)

        dense_gray = Image.open(output_path).convert("L")
        dense_gray = dense_gray.resize((w, h), Image.NEAREST)
        # remove image_path and output_path
        os.remove(image_path)
        os.remove(output_path)


        return dense_gray


if __name__ == '__main__':
    pass
