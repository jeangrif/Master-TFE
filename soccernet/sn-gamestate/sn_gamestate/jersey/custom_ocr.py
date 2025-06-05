import pandas as pd
import torch
import logging
from pathlib import Path
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from tracklab.utils.collate import default_collate, Unbatchable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from OCR.jersey_recognition_module import LegibilityModule, PoseDetector, PARSeqModule
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../OCR/parseq")))
log = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor

class CustomOCR(DetectionLevelModule):
    input_columns = []
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size

        root = Path(__file__).resolve().parents[3]  # remonte depuis soccernet/jersey/custom_ocr.py → racine
        self.legibility_model_path = str((root / cfg.legibility_model_path).resolve())
        self.pose_model_path = str((root / cfg.pose_model_path).resolve())
        self.parseq_path = str((root / cfg.parseq_path).resolve())

        self.legibility_module = LegibilityModule(self.legibility_model_path, device)
        self.pose_detector = PoseDetector(self.pose_model_path)
        self.ocr_module = PARSeqModule(self.parseq_path, device)

    def no_jersey_number(self):
        return None, 0.0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        # Découpe le crop complet de la bbox YOLO
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        return {"img": Unbatchable([crop])}
    """"
    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []

        for img in batch['img']:
            img_np = img.cpu().numpy()

            # Étape 1 : lisibilité
            crops, scores = self.legibility_module.run_from_crop(img_np)
            # Filtrage
            legible_crops = [crop for crop, score in zip(crops, scores) if score > 0.5]
            legible_scores = [score for score in scores if score > 0.5]

            if not legible_crops:
                number, score = self.no_jersey_number()
                jersey_number_detection.append(number)
                jersey_number_confidence.append(score)
                continue

            # Étape 2 : extraction du torse
            torso_crops, _ = self.pose_detector.infer(legible_crops)

            if not torso_crops:
                number, score = self.no_jersey_number()
                jersey_number_detection.append(number)
                jersey_number_confidence.append(score)
                continue

            # Étape 3 : OCR sur les torses
            numbers = self.ocr_module.infer(torso_crops)

            if not numbers:
                number, score = self.no_jersey_number()
                jersey_number_detection.append(number)
                jersey_number_confidence.append(score)
                continue

            # Prend le 1er numéro lisible (comme dans le main)
            jersey_number_detection.append(numbers[0])
            jersey_number_confidence.append(legible_scores[0])

        detections["jersey_number_detection"] = jersey_number_detection
        detections["jersey_number_confidence"] = jersey_number_confidence

        return detections
    """

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []

        # === Étape 1 : lisibilité en parallèle ===
        with ThreadPoolExecutor(max_workers=8) as executor:
            legibility_outputs = list(executor.map(self.legibility_module.run_from_crop, batch['img']))

        for i, (crops, scores) in enumerate(legibility_outputs):
            # Filtrage des bons crops
            legible_crops = [crop for crop, score in zip(crops, scores) if score > 0.5]
            legible_scores = [score for score in scores if score > 0.5]

            if not legible_crops:
                jersey_number_detection.append(None)
                jersey_number_confidence.append(0.0)
                continue

            # === Étape 2 : pose en parallèle sur les crops lisibles ===
            with ThreadPoolExecutor(max_workers=8) as executor:
                pose_results = list(executor.map(self.pose_detector.infer_single, legible_crops))

            torso_crops = [crop for crop, _ in pose_results if crop is not None]

            if not torso_crops:
                jersey_number_detection.append(None)
                jersey_number_confidence.append(0.0)
                continue

            # === Étape 3 : OCR PARSeq en parallèle ===
            with ThreadPoolExecutor(max_workers=8) as executor:
                ocr_results = list(executor.map(self.ocr_module.infer, torso_crops))

            if not ocr_results or not ocr_results[0]:
                jersey_number_detection.append(None)
                jersey_number_confidence.append(0.0)
                continue

            jersey_number_detection.append(ocr_results[0])
            jersey_number_confidence.append(legible_scores[0])

        detections["jersey_number_detection"] = jersey_number_detection
        detections["jersey_number_confidence"] = jersey_number_confidence

        return detections

