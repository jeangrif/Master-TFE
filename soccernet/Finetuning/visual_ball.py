import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =============== CONFIGURATION ===============
PRED_PATH = "predictions_adaptatives.json"
FRAME_FOLDER = "/Users/jeangrifnee/PycharmProjects/soccernet/data/SoccerNetGS/valid/SNGS-023/img1"
OUTPUT_VIDEO_PATH = "video_balle_interpolee.mp4"
BALL_CLASS_ID = 1
PLAYER_CLASS_ID = 0
CONF_THRESHOLD_DRAW = 0.4
POINT_RADIUS = 8
TRAIL_LENGTH = 15
FRAME_RATE = 25.0

# =============== UTILS ===============
def interpolate_bbox(box1, box2, alpha):
    return [int((1 - alpha) * b1 + alpha * b2) for b1, b2 in zip(box1, box2)]

# =============== CHARGEMENT PRÉDICTIONS ===============
with open(PRED_PATH, "r") as f:
    predictions = json.load(f)

frame_files = sorted(Path(FRAME_FOLDER).glob("*.jpg"), key=lambda x: int(x.stem))
frame_ids = [f.stem for f in frame_files]

# Extraction des bbox de balle + stockage de tous les preds pour les players
bboxes_by_frame = {}
bbox_confidences = {}

for frame_id in frame_ids:
    preds = predictions.get(frame_id, [])
    ball_preds = [p for p in preds if p["class_id"] == BALL_CLASS_ID]
    if ball_preds:
        best = max(ball_preds, key=lambda p: p["conf"])
        bboxes_by_frame[frame_id] = best["bbox"]
        bbox_confidences[frame_id] = best["conf"]

# =============== INTERPOLATION DES TROUS DE BALLE ===============
frame_indices_with_ball = [i for i, f in enumerate(frame_ids) if f in bboxes_by_frame]

for i in range(len(frame_indices_with_ball) - 1):
    start_idx = frame_indices_with_ball[i]
    end_idx = frame_indices_with_ball[i + 1]
    gap = end_idx - start_idx
    if gap <= 1:
        continue
    box1 = bboxes_by_frame[frame_ids[start_idx]]
    box2 = bboxes_by_frame[frame_ids[end_idx]]
    for j in range(1, gap):
        inter_idx = start_idx + j
        alpha = j / gap
        inter_box = interpolate_bbox(box1, box2, alpha)
        bboxes_by_frame[frame_ids[inter_idx]] = inter_box
        bbox_confidences[frame_ids[inter_idx]] = 0.0  # On sait que c'est interpolé

# =============== GÉNÉRATION VIDÉO ===============
first_frame = cv2.imread(str(frame_files[0]))
height, width = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, (width, height))

trail = []

for frame_path in tqdm(frame_files, desc="🎞️ Génération vidéo"):
    frame = cv2.imread(str(frame_path))
    frame_id = frame_path.stem

    # Players : affichage direct si conf > seuil
    preds = predictions.get(frame_id, [])
    for pred in preds:
        if pred["class_id"] == PLAYER_CLASS_ID and pred["conf"] >= CONF_THRESHOLD_DRAW:
            x1, y1, x2, y2 = pred["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 🔵 blue box

    # Balle : mise à jour du trail
    if frame_id in bboxes_by_frame:
        x1, y1, x2, y2 = bboxes_by_frame[frame_id]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        trail.append((cx, cy))
        if len(trail) > TRAIL_LENGTH:
            trail.pop(0)

        if bbox_confidences.get(frame_id, 0.0) >= CONF_THRESHOLD_DRAW:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 🔴 red box

    # Trail : cercles dégressifs
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / len(trail)
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), POINT_RADIUS, (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    video_writer.write(frame)

video_writer.release()
print(f"✅ Vidéo générée avec succès : {OUTPUT_VIDEO_PATH}")
