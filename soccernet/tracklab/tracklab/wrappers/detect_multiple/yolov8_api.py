import os
import torch
import pandas as pd
import numpy as np
from typing import Any
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from ultralytics.engine.results import Boxes
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from tracklab.utils.coordinates import ltrb_to_ltwh

import logging

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOv8(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        print("YOLOV8")
        print(device)

        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
        self.id = 0

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        #print(f"MPS dispo : {torch.backends.mps.is_available()}")
        #rint(f"MPS compilé : {torch.backends.mps.is_built()}")
        #print(f"[DEBUG] Mémoire MPS avant traitement : {torch.mps.current_allocated_memory()} bytes")
        results_by_image = self.model.predict(source=images, device="mps",verbose=False)
        #print(f"[DEBUG] Mémoire MPS après traitement : {torch.mps.current_allocated_memory()} bytes")
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            #print("coucou on est là)")
            #print(f"[DEBUG] results.boxes.xyxy device: {results.boxes.xyxy.device}")
            #print(f"[DEBUG] results.boxes.cls device: {results.boxes.cls.device}")
            #print(f"[DEBUG] results.boxes.conf device: {results.boxes.conf.device}")
            for bbox in results.boxes.cpu().numpy():
                # check for `person` class
                #print(f"[DEBUG] bbox type: {type(bbox)} | contenu: {bbox}")
                if not isinstance(bbox, Boxes):
                    print(f"[⚠️ WARNING] bbox inattendu ! Type: {type(bbox)} | Contenu: {bbox}")
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    xyxy = bbox.xyxy[0]

                    # Vérif sur bbox.xyxy elle-même
                    if (
                            not isinstance(xyxy, (np.ndarray, list, tuple))
                            or len(xyxy) != 4
                            or np.any(np.isnan(xyxy))
                            or xyxy[2] <= xyxy[0]
                            or xyxy[3] <= xyxy[1]
                    ):
                        print("\n" + "=" * 80)
                        print(f"[🛑 YOLO OUTPUT INVALID] bbox.xyxy malformée : {xyxy}")
                        print(f"[INFO] video_id={metadata.video_id} | image_id={metadata.name}")
                        print("=" * 80 + "\n")
                        quit()

                    # Conversion bbox en ltwh
                    ltwh = ltrb_to_ltwh(xyxy, shape)

                    if (
                            not isinstance(ltwh, (np.ndarray, list, tuple))
                            or len(ltwh) != 4
                            or np.any(np.isnan(ltwh))
                            or ltwh[2] <= 0
                            or ltwh[3] <= 0
                    ):
                        print("\n" + "=" * 80)
                        print(f"[🛑 INVALID BBOX_LTWH] Conversion ltrb_to_ltwh a foiré : {ltwh}")
                        print(f"[INFO] video_id={metadata.video_id} | image_id={metadata.name}")
                        print(f"[DEBUG] bbox.xyxy utilisé : {xyxy}")
                        print("=" * 80 + "\n")
                        quit()

                    # Tout est bon, on ajoute la détection
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltwh,
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1



        del results_by_image  # Supprime les résultats du modèle pour éviter l'accumulation
        torch.mps.empty_cache()
        return detections
