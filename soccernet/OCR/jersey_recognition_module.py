# jersey_recognition_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from parseq.strhub.data.module import SceneTextDataModule
import os
import numpy as np
def _get_model_class(key):
    if 'parseq' in key.lower():
        from parseq.strhub.models.parseq.system import PARSeq as ModelClass
    else:
        raise ValueError(f"Unknown model key: {key}")
    return ModelClass

def load_from_checkpoint(checkpoint_path: str, **kwargs):
    ModelClass = _get_model_class("parseq")
    model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model
class PARSeqModule:
    def __init__(self, checkpoint_path, device="cpu"):

        self.device = device
        self.model = load_from_checkpoint(checkpoint_path).eval().to(device)
        self.transform = SceneTextDataModule.get_transform(img_size=(32, 128))

    @torch.no_grad()
    def infer(self, crops):
        results = []
        for crop in crops:
            image = self.transform(crop).unsqueeze(0).to(self.device)
            logits = self.model(image)
            probs = logits.softmax(-1)
            preds, _ = self.model.tokenizer.decode(probs)
            results.append(preds[0])
        return results


class PoseDetector:
    def __init__(self, pose_model_path):
        from ultralytics import YOLO
        self.model = YOLO(pose_model_path)
        if torch.backends.mps.is_available():
            self.model.to("mps")

        # Indices des keypoints pour COCO (ou ajuster selon modèle utilisé)
        self.keypoints_map = {
            "left_shoulder": 6,
            "right_shoulder": 7,
            "left_hip": 12,
            "right_hip": 13
        }

    def extract_torso_crop(self, image, keypoints):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        try:
            pts = [keypoints[i] for i in self.keypoints_map.values()]
            valid_pts = [(int(x), int(y)) for x, y, conf in pts if conf > 0.2]

            if len(valid_pts) < 2:
                #print("❌ Pas assez de keypoints valides :", valid_pts)
                return None, None

            # Fallback général : bbox des points valides
            x_coords = [p[0] for p in valid_pts]
            y_coords = [p[1] for p in valid_pts]
            x_min = max(0, min(x_coords))
            x_max = min(w, max(x_coords))
            y_min = max(0, min(y_coords))
            y_max = min(h, max(y_coords))

            # Si on a bien les 4 points, on spécialise pour un torse "idéal"
            indices = self.keypoints_map
            required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            if all(keypoints[indices[k]][2] > 0.2 for k in required):
                sx1, sy1 = keypoints[indices["left_shoulder"]][:2]
                sx2, sy2 = keypoints[indices["right_shoulder"]][:2]
                hx1, hy1 = keypoints[indices["left_hip"]][:2]
                hx2, hy2 = keypoints[indices["right_hip"]][:2]

                x_min = max(0, int(min(sx1, sx2, hx1, hx2)))
                x_max = min(w, int(max(sx1, sx2, hx1, hx2)))
                y_min = max(0, int(min(sy1, sy2)))
                y_max = min(h, int(max(hy1, hy2)))
                #print("✅ Torse complet détecté avec épaules/hanches")

            if x_max - x_min <10 or y_max - y_min < 10:
                #print("❌ Torse crop trop petit :", (x_min, y_min, x_max, y_max))
                return None, None

            crop = image.crop((x_min, y_min, x_max, y_max))


            return crop, (x_min, y_min, x_max, y_max)

        except Exception as e:
            #print("⚠️ Erreur dans extract_torso_crop :", e)
            return None, None

    def infer(self, crops):
        torso_crops = []
        torso_boxes = []
        for i, crop in enumerate(crops):

            result = self.model.predict(crop, verbose=False)[0]
            if result.keypoints is None or len(result.keypoints.data) == 0:
                crop.show()
                continue

            keypoints = result.keypoints.data[0].cpu().numpy()
            torso_crop, box = self.extract_torso_crop(crop, keypoints)
            if torso_crop is not None:
                torso_crops.append(torso_crop)
                torso_boxes.append(box)
        return torso_crops, torso_boxes

    def infer_single(self, crop):
        import numpy as np
        from PIL import Image

        # Assurer un format numpy RGB
        if isinstance(crop, torch.Tensor):
            crop = crop.cpu().numpy()
        elif isinstance(crop, Image.Image):
            crop = np.array(crop)

        # Ajouter une dimension batch si nécessaire
        if crop.ndim == 4 and crop.shape[0] == 1:
            crop = crop[0]  # retire la batch dim inutile

        result = self.model.predict(crop, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None, None

        keypoints = result.keypoints.data[0].cpu().numpy()
        return self.extract_torso_crop(crop[0], keypoints)


class LegibilityClassifier34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model_ft(x)
        x = torch.sigmoid(x)
        return x


class LegibilityModule:
    def __init__(self, weights_path: str, device="cpu"):
        self.device = device
        self.model = LegibilityClassifier34().to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_yolo_bboxes(self, label_path):
        bboxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = map(float, parts)
                bboxes.append((cx, cy, w, h))
        return bboxes

    def crop_bbox(self, image, bbox):
        cx, cy, w, h = bbox
        img_w, img_h = image.size
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        return image.crop((x1, y1, x2, y2))

    def run(self, image_path: str, label_path: str):
        image = Image.open(image_path).convert("RGB")
        bboxes = self.load_yolo_bboxes(label_path)

        crops = [self.crop_bbox(image, bbox) for bbox in bboxes]
        inputs = torch.stack([self.transform(crop) for crop in crops]).to(self.device)

        with torch.no_grad():
            preds = self.model(inputs).squeeze().cpu().numpy()

        return crops, preds.tolist()

    @torch.no_grad()
    def run_from_crop(self, crop_input):
        """
        Applique le module de lisibilité sur une image cropée (Tensor, np.array ou PIL).
        Retourne une liste de crops (1 seul ici) + scores de lisibilité.
        """
        from PIL import Image
        import numpy as np

        # === Forcer à numpy si c’est un Tensor ===
        if isinstance(crop_input, torch.Tensor):
            crop_np = crop_input.cpu().numpy()
        elif isinstance(crop_input, np.ndarray):
            crop_np = crop_input
        else:
            print("[Legibility] Erreur : entrée non reconnue")
            return [], []

        try:
            img_pil = Image.fromarray(crop_np)
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[Legibility] Erreur de transformation : {e}")
            return [], []

        score = torch.sigmoid(self.model(img_tensor)).item()
        return [crop_np], [score]
