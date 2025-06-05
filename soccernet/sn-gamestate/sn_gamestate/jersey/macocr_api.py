import pandas as pd
import torch
import logging
import numpy as np
import tempfile
from PIL import Image
from ocrmac.ocrmac import text_from_image

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

log = logging.getLogger(__name__)


class MACOCR(DetectionLevelModule):
    input_columns = []
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)

        log.debug("🔧 MACOCR initialized with batch_size=%d", batch_size)

    def no_jersey_number(self):
        return [None, None, 0]

    def ocr_from_np(self, np_img):
        image = Image.fromarray(np_img)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            result = text_from_image(tmp.name)

        # Gestion des cas où c'est une liste ou vide
        if isinstance(result, list):
            result = result[0] if result else ""

        # On nettoie et on retourne une string (ou chaîne vide)
        return str(result).strip()

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []

        log.debug("🚀 MACOCR process() called on %d images", len(batch['img']))

        for i, img in enumerate(batch['img']):
            np_img = img.cpu().numpy()
            text = self.ocr_from_np(np_img)

            if not text:
                jn = self.no_jersey_number()
                log.debug("❌ No text detected for image %d", i)
            else:
                try:
                    int(text)
                    jn = [None, text, 1.0]
                    log.debug("✅ Detected jersey number: %s", text)
                except ValueError:
                    jn = self.no_jersey_number()
                    log.debug("❌ Non-numeric OCR result: %s", text)

            jersey_number_detection.append(jn[1])
            jersey_number_confidence.append(jn[2])

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence

        return detections
