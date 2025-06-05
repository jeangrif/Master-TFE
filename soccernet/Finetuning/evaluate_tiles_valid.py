import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from yolo_model import YOLOModel
from resize import ResizeWithPadding
import matplotlib.pyplot as plt
import json

def save_predictions_to_json(predictions, path):
    with open(path, "w") as f:
        json.dump(predictions, f)

def load_predictions_from_json(path):
    with open(path, "r") as f:
        return json.load(f)

import torch
import numpy as np
from ultralytics.utils.metrics import bbox_iou

def evaluate_predictions(predictions, dataloader, conf_threshold=0.25, iou_threshold=0.5):
    class_names = {0: "Person", 1: "Ball"}
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_names}

    # On crée un dictionnaire image_id → prédiction
    pred_dict = {pred["image_id"]: pred for pred in predictions}

    # Parcours du dataloader
    for images, gt_labels_batch, image_ids, sizes in dataloader:
        for img, labels, image_id, (img_w, img_h) in zip(images, gt_labels_batch, image_ids, sizes):
            if image_id not in pred_dict:
                continue

            pred_sample = pred_dict[image_id]

            for class_id in class_names:
                # Ground truth : bbox converties en absolu
                gt_boxes = []
                for ann in labels:
                    if int(ann[0]) != class_id:
                        continue
                    xc, yc, w, h = ann[1:]
                    x1 = (xc - w / 2) * img_w
                    y1 = (yc - h / 2) * img_h
                    x2 = (xc + w / 2) * img_w
                    y2 = (yc + h / 2) * img_h
                    gt_boxes.append([x1, y1, x2, y2])
                gt_boxes = torch.tensor(gt_boxes)

                # Prédictions filtrées par classe et confiance
                pred_boxes = []
                scores = []
                for box, score, cls in zip(pred_sample["boxes"], pred_sample["scores"], pred_sample["classes"]):
                    if score >= conf_threshold and int(cls) == class_id:
                        pred_boxes.append(box)
                        scores.append(score)

                matched_gt = set()
                for box, score in zip(pred_boxes, scores):
                    if len(gt_boxes) == 0:
                        stats[class_id]["fp"] += 1
                        continue
                    ious = bbox_iou(torch.tensor([box]), gt_boxes)
                    max_iou, idx = ious.max(1)
                    if max_iou >= iou_threshold and idx.item() not in matched_gt:
                        stats[class_id]["tp"] += 1
                        matched_gt.add(idx.item())
                    else:
                        stats[class_id]["fp"] += 1

                stats[class_id]["fn"] += len(gt_boxes) - len(matched_gt)

    # 🔢 Calculs finaux
    for class_id, res in stats.items():
        tp, fp, fn = res["tp"], res["fp"], res["fn"]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"📊 {class_names[class_id]} → TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")



# === DATASET CLASSE ===
class YOLOTileDatasetWithLabels(Dataset):
    def __init__(self, root_dir, split="valid", transforms=None):
        self.image_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(".jpg")
        ])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        label_path = os.path.join(self.label_dir, image_id + ".txt")
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    labels.append(parts)  # [cls, xc, yc, w, h]

        if self.transforms:
            image = self.transforms(image)

        return image, labels, image_id, (width, height)


# === COLLATE ===
def collate_fn(batch):
    images, labels, image_ids, sizes = zip(*batch)
    return list(images), list(labels), list(image_ids), list(sizes)

# === CONFIGURATION ===
MODEL_PATH = "first_test_finetuning_whith_skip_frame5/fine_tuned/weights/best.pt"
DATA_ROOT = "../datasetforeval"
SPLIT = "valid"
BATCH_SIZE = 8
IMG_SIZE = 640

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
REGENERATE_PREDICTIONS = False
PREDICTION_PATH = f"predictions_{SPLIT}.json"
# === TRANSFORM ===
transform = transforms.Compose([
    ResizeWithPadding((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = YOLOTileDatasetWithLabels(DATA_ROOT, SPLIT, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === CHARGEMENT MODELE ===
model = YOLOModel(MODEL_PATH)
print(f"🎯 Modèle chargé : {MODEL_PATH}")

# === INFERENCE ou CHARGEMENT ===
if REGENERATE_PREDICTIONS or not os.path.exists(PREDICTION_PATH):
    print("🔎 Inference YOLO...")
    predictions, fps = model.inference(dataloader, conf_threshold=CONF_THRESHOLD)

    preds_json = []
    for p in predictions:
        preds_json.append({
            "image_id": p["image_id"],
            "boxes": p["boxes"].tolist(),
            "scores": p["scores"].tolist(),
            "classes": p["classes"].tolist()
        })

    with open(PREDICTION_PATH, "w") as f:
        json.dump(preds_json, f)
    print(f"✅ Prédictions sauvegardées → {PREDICTION_PATH}")
else:
    print(f"📂 Chargement des prédictions depuis {PREDICTION_PATH}")
    with open(PREDICTION_PATH, "r") as f:
        predictions = json.load(f)