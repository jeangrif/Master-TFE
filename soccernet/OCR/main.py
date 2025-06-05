import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import cv2
import torch
import numpy as np
from jersey_recognition_module import LegibilityModule, PoseDetector
from jersey_recognition_module import PARSeqModule
import sys
import os
from pathlib import Path
from rich.progress import Progress
from tqdm import tqdm
sys.path.append(os.path.abspath("parseq"))
from rich.progress import track

if __name__ == "__main__":
    # === Chemins ===
    base_image_dir = Path("/Users/jeangrifnee/PycharmProjects/soccernet/dataset_person_only/images/valid")
    base_label_dir = Path("/Users/jeangrifnee/PycharmProjects/soccernet/dataset_person_only/labels/valid")
    weights_path = "/Users/jeangrifnee/PycharmProjects/soccernet/OCR/legibility_resnet34_soccer_20240215.pth"
    pose_model_path = "/Users/jeangrifnee/PycharmProjects/soccernet/OCR/yolo11l-pose.pt"  # <- adapte ça
    ckpt_path = "/Users/jeangrifnee/PycharmProjects/soccernet/OCR/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"

    # === Initialisation des modules ===
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    legibility_module = LegibilityModule(weights_path=weights_path, device=device)
    pose_detector = PoseDetector(pose_model_path=pose_model_path)
    str_module = PARSeqModule(checkpoint_path=ckpt_path, device=device)

    clip_id = 20210001
    image_paths = [base_image_dir / f"{clip_id + i}.jpg" for i in range(750)]
    label_paths = [base_label_dir / f"{clip_id + i}.txt" for i in range(750)]


    results_per_frame = []

    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=750, desc="🔢 Traitement des frames"):
        if not image_path.exists() or not label_path.exists():
            results_per_frame.append([])

            continue

        crops, scores = legibility_module.run(str(image_path), str(label_path))
        legible_crops = [crop for crop, score in zip(crops, scores) if score > 0.5]
        legible_scores = [score for score in scores if score > 0.5]

        torso_crops, _ = pose_detector.infer(legible_crops)
        numbers = str_module.infer(torso_crops)

        frame_results = [{"image_path": str(image_path), "number": n, "score": s}
                         for n, s in zip(numbers, legible_scores)]
        results_per_frame.append(frame_results)

    # === Résumé final (optionnel) ===
    print("\n📋 Résultats par frame :")
    for i, frame in enumerate(results_per_frame):
        image_id = 20210001 + i  # correspond à l'ID réel de la frame
        if frame:
            print(f"\n🖼️ Frame {image_id} :")
            for res in frame:
                print(f" - Numéro : {res['number']} (score {res['score']:.2f})")