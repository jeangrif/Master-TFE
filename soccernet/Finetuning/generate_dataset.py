import os
import shutil
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from dataloader import SoccerNetDataset

# 📁 CONFIGURATION
SOURCE_ROOT = "../data/SoccerNetGS"
SUBSET = "valid"
SPLIT = "valid"
MAX_IMAGES = None  # None = tout prendre
TILE_SIZE = 384
TILE_OVERLAP = 128
BALL_CLASS_ID = 1
PERSON_CLASS_ID = 0
TILE_BALL_RATIO = 0.85  # pour "ball-only"

# 📁 OUTPUT
DEST_BALL_DIR = "../dataset_ball_only"
DEST_PERSON_DIR = "../dataset_person_only"
os.makedirs(f"{DEST_BALL_DIR}/images/{SPLIT}", exist_ok=True)
os.makedirs(f"{DEST_BALL_DIR}/labels/{SPLIT}", exist_ok=True)
os.makedirs(f"{DEST_PERSON_DIR}/images/{SPLIT}", exist_ok=True)
os.makedirs(f"{DEST_PERSON_DIR}/labels/{SPLIT}", exist_ok=True)

# 📦 DATASET
dataset = SoccerNetDataset(SOURCE_ROOT, subset=SUBSET, skip_step=15)
print(f"✅ Total frames chargées : {len(dataset)} (limite : {MAX_IMAGES})")

# =============== BALL-ONLY ===============
def tile_and_save_ball_only(img, labels, image_id, width, height):
    img_np = np.array(img)
    tiles_x = range(0, width, TILE_SIZE - TILE_OVERLAP)
    tiles_y = range(0, height, TILE_SIZE - TILE_OVERLAP)

    tiles_with_ball = []
    tiles_without_ball = []

    for x in tiles_x:
        for y in tiles_y:
            x_end = min(x + TILE_SIZE, width)
            y_end = min(y + TILE_SIZE, height)
            tile = img_np[y:y_end, x:x_end]
            tile_w, tile_h = x_end - x, y_end - y

            tile_labels = []
            for cls, xc, yc, w, h in labels:
                abs_x = xc * width
                abs_y = yc * height
                abs_w = w * width
                abs_h = h * height

                bbox_x1 = abs_x - abs_w / 2
                bbox_y1 = abs_y - abs_h / 2
                bbox_x2 = abs_x + abs_w / 2
                bbox_y2 = abs_y + abs_h / 2

                inter_x1 = max(bbox_x1, x)
                inter_y1 = max(bbox_y1, y)
                inter_x2 = min(bbox_x2, x_end)
                inter_y2 = min(bbox_y2, y_end)

                inter_w = inter_x2 - inter_x1
                inter_h = inter_y2 - inter_y1
                if inter_w <= 0 or inter_h <= 0:
                    continue

                inter_area = inter_w * inter_h
                original_area = abs_w * abs_h
                if inter_area / original_area < 0.7:
                    continue

                new_xc = ((inter_x1 + inter_x2) / 2 - x) / tile_w
                new_yc = ((inter_y1 + inter_y2) / 2 - y) / tile_h
                new_w = inter_w / tile_w
                new_h = inter_h / tile_h
                tile_labels.append([cls, new_xc, new_yc, new_w, new_h])

            if len(tile_labels) == 0:
                continue

            tile_image_id = f"{image_id}_tile_{x}_{y}"
            tile_img = Image.fromarray(tile)
            label_path = os.path.join(DEST_BALL_DIR, "labels", SPLIT, f"{tile_image_id}.txt")
            img_path = os.path.join(DEST_BALL_DIR, "images", SPLIT, f"{tile_image_id}.jpg")

            ball_labels = [l for l in tile_labels if int(l[0]) == BALL_CLASS_ID]
            if ball_labels:
                # 🎯 On garde uniquement les labels ball
                tile_img.save(img_path)
                with open(label_path, "w") as f:
                    for l in ball_labels:
                        f.write(" ".join([f"{x:.6f}" for x in l]) + "\n")
                tiles_with_ball.append(tile_image_id)
            else:
                tiles_without_ball.append((tile_img, img_path, label_path, tile_labels))

    return tiles_with_ball, tiles_without_ball

tiles_with_ball_total = 0
tiles_no_ball_buffer = []

for i in tqdm(range(len(dataset) if MAX_IMAGES is None else min(len(dataset), MAX_IMAGES)), desc="🔄 Génération Ball-only"):
    img, labels, image_id, (width, height) = dataset[i]
    if any(lbl[0] == BALL_CLASS_ID for lbl in labels):
        tiles_ball, tiles_no_ball = tile_and_save_ball_only(img, labels, image_id, width, height)
        tiles_with_ball_total += len(tiles_ball)
        tiles_no_ball_buffer.extend(tiles_no_ball)

# ➕ Ajout de tiles sans balle pour atteindre le ratio
nb_total_target = int(tiles_with_ball_total / TILE_BALL_RATIO)
nb_no_ball_to_add = max(0, nb_total_target - tiles_with_ball_total)
random.shuffle(tiles_no_ball_buffer)
selected = tiles_no_ball_buffer[:nb_no_ball_to_add]

for tile_img, img_path, label_path, tile_labels in selected:
    tile_img.save(img_path)
    with open(label_path, "w") as f:
        for l in tile_labels:
            if int(l[0]) == BALL_CLASS_ID:
                f.write(" ".join([f"{x:.6f}" for x in l]) + "\n")

print(f"\n📦 Résumé final :")
print(f"🎯 Ball-only : {tiles_with_ball_total} tiles avec balle + {len(selected)} sans balle")
print(f"   → Ratio balle ≈ {100 * tiles_with_ball_total / (tiles_with_ball_total + len(selected)):.2f}%")
"""
# =============== PERSON-ONLY ===============
print("\n👥 Génération du dataset Person-only (entière sans tiling)")
for i in tqdm(range(len(dataset)), desc="👥 Export Person"):
    img, labels, image_id, _ = dataset[i]
    img_path_src = dataset.data[i]["image_path"]
    img_path_dst = os.path.join(DEST_PERSON_DIR, "images", SPLIT, f"{image_id}.jpg")
    shutil.copy(img_path_src, img_path_dst)

    person_labels = [lbl for lbl in labels if int(lbl[0]) == PERSON_CLASS_ID]
    if person_labels:
        with open(os.path.join(DEST_PERSON_DIR, "labels", SPLIT, f"{image_id}.txt"), "w") as f:
            for label in person_labels:
                f.write(" ".join([f"{x:.6f}" for x in label]) + "\n")

print(f"\n✅ Person-only : {len(dataset)} images copiées avec uniquement les annotations de joueur.")
"""