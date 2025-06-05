import os
import shutil
from tqdm import tqdm
from dataloader import SoccerNetDataset  # adapte si ton fichier a un autre nom
import numpy as np
from PIL import Image
# ⚙️ Configuration
SOURCE_ROOT = "../data/SoccerNetGS"      # dossier contenant train/ valid/ test/
DEST_ROOT = "../fullball"     # où tu veux créer images/ et labels/
SUBSET = "valid"                         # "valid" ou "valid" ou "test"
SPLIT = "valid"                          # le nom du split YOLO : "train" ou "val"
MAX_IMAGES = 150                      # pour test rapide
TILING = True  # Active le tiling conditionnel pour la balle
TILE_SIZE = 384
TILE_OVERLAP = 128  # Tu peux mettre 64 si tu veux un petit overlap
# 📦 Chargement du dataset sans transform
dataset = SoccerNetDataset(SOURCE_ROOT, subset=SUBSET,skip_step=10)
TILE_BALL_RATIO = 0.8
# 📁 Création des dossiers YOLO
image_output_dir = os.path.join(DEST_ROOT, "images", SPLIT)
label_output_dir = os.path.join(DEST_ROOT, "labels", SPLIT)
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def tile_and_save(img, labels, image_id, width, height, img_dir, label_dir, tile_size=640, overlap=0):
    img_np = np.array(img)
    tiles_x = range(0, width, tile_size - overlap)
    tiles_y = range(0, height, tile_size - overlap)

    tiles_created = 0
    tiles_with_ball = 0
    tiles_no_ball_buffer = []  # Pour stocker les tiles sans ball à filtrer plus tard

    for x in tiles_x:
        for y in tiles_y:
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
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
                visibility_ratio = inter_area / original_area

                if visibility_ratio < 0.7:
                    continue

                new_xc = ((inter_x1 + inter_x2) / 2 - x) / tile_w
                new_yc = ((inter_y1 + inter_y2) / 2 - y) / tile_h
                new_w = inter_w / tile_w
                new_h = inter_h / tile_h

                tile_labels.append([
                    cls,
                    new_xc,
                    new_yc,
                    new_w,
                    new_h
                ])

            if len(tile_labels) == 0:
                continue

            tiles_created += 1
            tile_image_id = f"{image_id}_tile_{x}_{y}"
            tile_path = os.path.join(img_dir, f"{tile_image_id}.jpg")
            label_path = os.path.join(label_dir, f"{tile_image_id}.txt")
            tile_img = Image.fromarray(tile)

            if any(lbl[0] == 1 for lbl in tile_labels):
                # 🎯 contient une balle → on sauvegarde tout de suite
                tile_img.save(tile_path)
                with open(label_path, "w") as f:
                    for label in tile_labels:
                        f.write(" ".join([f"{x:.6f}" for x in label]) + "\n")
                tiles_with_ball += 1
            else:
                # Pas de ball → on le stocke temporairement pour filtrage
                tiles_no_ball_buffer.append((tile_img, tile_path, label_path, tile_labels))

    return tiles_created, tiles_with_ball, tiles_no_ball_buffer





total_tiles = 0
tiles_with_ball = 0
tiles_no_ball_buffer = []
for i in tqdm(range(min(len(dataset), MAX_IMAGES)), desc=f"🔄 Conversion {SUBSET} → {SPLIT}"):
    img, labels, image_id, (width, height) = dataset[i]

    if TILING and dataset.contains_small_ball(labels, width, height):
        tiles_gen, tiles_ball, tiles_no_ball  = tile_and_save(img, labels, image_id, width, height,
                                              image_output_dir, label_output_dir,
                                              TILE_SIZE, TILE_OVERLAP)
        tiles_with_ball += tiles_ball
        tiles_no_ball_buffer.extend(tiles_no_ball)
    else:
        image_name = f"{image_id}.jpg"
        img_path_src = dataset.data[i]["image_path"]
        img_path_dst = os.path.join(image_output_dir, image_name)
        shutil.copy(img_path_src, img_path_dst)

        label_path = os.path.join(label_output_dir, f"{image_id}.txt")
        with open(label_path, "w") as f:
            for label in labels:
                f.write(" ".join([f"{x:.6f}" for x in label]) + "\n")


# ➕ Ajout des tiles sans balle pour atteindre le ratio TILE_BALL_RATIO
nb_total_tiles_target = int(tiles_with_ball / TILE_BALL_RATIO)
nb_no_ball_to_add = nb_total_tiles_target - tiles_with_ball

print(f"🧮 Objectif : {nb_total_tiles_target} tiles au total")
print(f"📊 À ajouter : {nb_no_ball_to_add} tiles sans balle sur {len(tiles_no_ball_buffer)} dispo")

import random
random.shuffle(tiles_no_ball_buffer)
selected_no_ball = tiles_no_ball_buffer[:nb_no_ball_to_add]

for tile_img, tile_path, label_path, tile_labels in selected_no_ball:
    tile_img.save(tile_path)
    with open(label_path, "w") as f:
        for label in tile_labels:
            f.write(" ".join([f"{x:.6f}" for x in label]) + "\n")

final_total_tiles = tiles_with_ball + len(selected_no_ball)
print(f"✅ Tiles finales : {final_total_tiles} (balle: {tiles_with_ball}, sans balle: {len(selected_no_ball)})")
print(f"🎯 Ratio final balle ≈ {100 * tiles_with_ball / final_total_tiles:.2f}%")