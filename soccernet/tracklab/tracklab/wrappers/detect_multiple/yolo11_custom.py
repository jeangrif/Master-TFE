import os
import torch
import pandas as pd
import numpy as np
from typing import Any
from ultralytics.engine.results import Boxes
from ultralytics import YOLO
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh

os.environ["YOLO_VERBOSE"] = "False"

import logging
log = logging.getLogger(__name__)

def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)

class YOLOv8Dual(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
        "ball",  # champ ajouté
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.id = 0

        self.player_model = YOLO(cfg.path_to_player_checkpoint)
        self.ball_model = YOLO(cfg.path_to_ball_checkpoint)

        if torch.backends.mps.is_available():
            self.player_model.to("mps")
            self.ball_model.to("mps")

        self.tile_size = cfg.tile_size if hasattr(cfg, "tile_size") else 384
        self.conf_thresh = cfg.min_confidence if hasattr(cfg, "min_confidence") else 0.3
        self.detect_ball = getattr(cfg, "detect_ball", True)

        # Dictionnaire de mémoire des positions précédentes de balle
        self.past_ball_positions = {}

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    def generate_all_tiles(self, image, overlap_ratio=0.25):
        h, w = image.shape[:2]
        stride = int(self.tile_size * (1 - overlap_ratio))
        tiles = []
        for y in range(0, h - self.tile_size + 1, stride):
            for x in range(0, w - self.tile_size + 1, stride):
                tile = image[y:y + self.tile_size, x:x + self.tile_size]
                tiles.append((tile, (x, y)))
        return tiles

    def generate_tile_centered_on_bbox(self, image, bbox):
        h, w = image.shape[:2]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        x0 = int(max(0, min(w - self.tile_size, cx - self.tile_size // 2)))
        y0 = int(max(0, min(h - self.tile_size, cy - self.tile_size // 2)))
        tile = image[y0:y0 + self.tile_size, x0:x0 + self.tile_size]
        return [(tile, (x0, y0))]

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        output = []

        player_results = self.player_model.predict(source=images, device="mps", verbose=False)

        for i, (image, shape, (_, metadata)) in enumerate(zip(images, shapes, metadatas.iterrows())):
            frame_detections = []
            image_id = metadata.name
            video_id = metadata.video_id

            # ---- JOUEURS ----
            for box in player_results[i].boxes.cpu():
                if box.cls == 0 and box.conf >= self.conf_thresh:
                    xyxy = box.xyxy[0].numpy()
                    ltwh = ltrb_to_ltwh(xyxy, shape)
                    frame_detections.append(pd.Series(dict(
                        image_id=image_id,
                        video_id=video_id,
                        category_id=0,
                        bbox_ltwh=ltwh,
                        bbox_conf=box.conf.item(),
                        ball=False,
                    ), name=self.id))
                    self.id += 1
            if self.detect_ball:
                # ---- BALLE ----
                # On regarde si une position précédente est disponible
                try:
                    prev_id = str(int(image_id) - 1)
                except:
                    prev_id = None

                found_ball = False
                if prev_id and prev_id in self.past_ball_positions:
                    bbox = self.past_ball_positions[prev_id]
                    if bbox is not None:
                        tiles = self.generate_tile_centered_on_bbox(image, bbox)
                    else:
                        tiles = self.generate_all_tiles(image)
                else:
                    tiles = self.generate_all_tiles(image)

                for tile, (x_off, y_off) in tiles:
                    results = self.ball_model(tile, verbose=False)[0]
                    for box in results.boxes.cpu():
                        if box.conf >= self.conf_thresh:
                            x1, y1, x2, y2 = box.xyxy[0].numpy()
                            global_box = np.array([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off])
                            ltwh = ltrb_to_ltwh(global_box, shape)
                            frame_detections.append(pd.Series(dict(
                                image_id=image_id,
                                video_id=video_id,
                                category_id=1,
                                bbox_ltwh=ltwh,
                                bbox_conf=box.conf.item(),
                                ball=True,
                            ), name=self.id))
                            self.past_ball_positions[image_id] = global_box
                            self.id += 1
                            found_ball = True

                if not found_ball:
                    self.past_ball_positions[image_id] = None

            output.extend(frame_detections)

        torch.mps.empty_cache()
        return output
