from sahi.slicing import slice_image
import cv2
from pathlib import Path
import os

def convert_bbox_yolo_to_absolute(x_center, y_center, width, height, img_w, img_h):
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h
    return x1, y1, x2, y2

def convert_bbox_absolute_to_yolo(x1, y1, x2, y2, tile_w, tile_h):
    x_center = ((x1 + x2) / 2) / tile_w
    y_center = ((y1 + y2) / 2) / tile_h
    width = (x2 - x1) / tile_w
    height = (y2 - y1) / tile_h
    return x_center, y_center, width, height

def slice_and_generate_yolo(split_name, base_input_dir="../output_yolo_dataset", base_output_dir="../output_yolo_dataset_sliced", slice_size=384, overlap=0.2):
    image_dir = Path(base_input_dir) / "images" / split_name
    label_dir = Path(base_input_dir) / "labels" / split_name
    out_image_dir = Path(base_output_dir) / split_name / "images"
    out_label_dir = Path(base_output_dir) / split_name / "labels"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    #print(f"[i] Dossier scanné : {image_dir}")
    #print(f"[i] Contenu : {[f.name for f in image_dir.iterdir()]}")
    for image_path in image_dir.glob("*.jpg"):
        #print(f"[→] Processing: {image_path.name}")
        label_path = label_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(image_path))
        img_h, img_w = img.shape[:2]

        # Lire les annotations
        with open(label_path, "r") as f:
            lines = f.read().splitlines()
        bboxes = []
        for line in lines:
            cls, x, y, w, h = map(float, line.strip().split())
            abs_box = convert_bbox_yolo_to_absolute(x, y, w, h, img_w, img_h)
            bboxes.append((cls, *abs_box))

        # Slicing de l'image
        result = slice_image(
            image=image_path.as_posix(),
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            out_ext=".jpg",
            out_dir=out_image_dir.as_posix(),
            return_result=True,
        )

        for slice_info in result["slice_image_result"]:
            tile_path = Path(slice_info.image_path)
            x_offset, y_offset = slice_info.start_x, slice_info.start_y
            tile_w, tile_h = slice_info.image_width, slice_info.image_height

            tile_labels = []

            for cls, x1, y1, x2, y2 in bboxes:
                # Vérifie si l'objet intersecte la tuile
                if x2 < x_offset or x1 > x_offset + tile_w:
                    continue
                if y2 < y_offset or y1 > y_offset + tile_h:
                    continue

                # Découpe la bbox aux limites de la tile (clipping)
                x1_clipped = max(x1, x_offset)
                y1_clipped = max(y1, y_offset)
                x2_clipped = min(x2, x_offset + tile_w)
                y2_clipped = min(y2, y_offset + tile_h)

                # Recaler dans la tile
                x1_rel = x1_clipped - x_offset
                y1_rel = y1_clipped - y_offset
                x2_rel = x2_clipped - x_offset
                y2_rel = y2_clipped - y_offset

                # Convertir en YOLO (normalisé)
                yolo_box = convert_bbox_absolute_to_yolo(x1_rel, y1_rel, x2_rel, y2_rel, tile_w, tile_h)

                # Vérifie taille minimale (évite les bboxes quasi-nulles)
                if yolo_box[2] < 0.01 or yolo_box[3] < 0.01:
                    continue

                tile_labels.append(f"{int(cls)} {' '.join(f'{v:.6f}' for v in yolo_box)}")

            # Sauver les labels
            out_label_path = out_label_dir / (tile_path.stem + ".txt")
            if tile_labels:
                with open(out_label_path, "w") as f:
                    f.write("\n".join(tile_labels))

        print(f"[✔] {image_path.name} → {len(result['slice_image_result'])} tiles générées avec annotations filtrées")

if __name__ == "__main__":
    for split in ["train", "valid"]:
        slice_and_generate_yolo(split_name=split)
