import os
import json
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class SoccerNetDataset(Dataset):
    SET_MAP = {"train": "1", "valid": "2", "test": "3", "challenge": "2"}

    def __init__(self, root_dir, subset, transforms=None, max_frames=None, skip_step=1):
        self.transforms = transforms
        self.data = []

        subset_digit = self.SET_MAP[subset]
        subset_dir = os.path.join(root_dir, subset)

        video_dirs = sorted(
            [os.path.join(subset_dir, d) for d in os.listdir(subset_dir)
             if os.path.isdir(os.path.join(subset_dir, d))],
            key=lambda x: int(x.split('-')[-1])
        )

        print(f"📁 Loading {subset.upper()} dataset...")

        for video_dir in tqdm(video_dirs, desc=f"🔍 Parsing {subset} videos"):
            if max_frames is not None and len(self.data) >= max_frames:
                break

            video_num = video_dir.split("-")[-1]
            img_dir = os.path.join(video_dir, "img1")
            labels_path = os.path.join(video_dir, "Labels-GameState.json")

            if not os.path.exists(labels_path):
                print(f"⚠️ Fichier JSON introuvable: {labels_path}")
                continue

            with open(labels_path, "r") as f:
                annotations_json = json.load(f)

            annotations_by_frame = {}
            for ann in annotations_json["annotations"]:
                image_id = ann["image_id"]
                frame_num = image_id[-4:]
                annotations_by_frame.setdefault(frame_num, []).append(ann)

            sorted_frames = sorted(annotations_by_frame.keys(), key=int)
            for idx, frame_num in enumerate(sorted_frames):
                if max_frames is not None and len(self.data) >= max_frames:
                    break
                if idx % skip_step != 0:
                    continue

                anns = annotations_by_frame[frame_num]
                image_filename = f"{frame_num.zfill(6)}.jpg"
                image_path = os.path.join(img_dir, image_filename)
                if os.path.exists(image_path):
                    self.data.append({
                        "image_path": image_path,
                        "annotations": anns,
                        "image_id": f"{subset_digit}{video_num}{frame_num}"
                    })

        print(f"✅ Total frames chargées : {len(self.data)} (limite : {max_frames})")

        nb_with_ball = 0
        total_players = 0
        for item in self.data:
            anns = item["annotations"]
            if any(ann["category_id"] == 4 for ann in anns):
                nb_with_ball += 1
            total_players += sum(1 for ann in anns if ann["category_id"] in [1, 2, 3])

        pct_ball = 100 * nb_with_ball / len(self.data)
        avg_players = total_players / len(self.data)

        print(f"🎯 {nb_with_ball} frames avec ballon sur {len(self.data)} frames totales ({pct_ball:.2f}%)")
        print(f"👥 Nombre moyen de joueurs/arbitres par frame : {avg_players:.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        width, height = img.size

        labels = []
        for ann in item["annotations"]:
            if "bbox_image" not in ann:
                continue
            if ann["category_id"] in [1, 2, 3]:
                class_id = 0
            elif ann["category_id"] == 4:
                class_id = 1
            else:
                continue

            bbox = ann["bbox_image"]
            labels.append([
                class_id,
                bbox["x_center"] / width,
                bbox["y_center"] / height,
                bbox["w"] / width,
                bbox["h"] / height
            ])

        if self.transforms:
            img = self.transforms(img)

        return img, labels, item["image_id"], (width, height)

    @staticmethod
    def contains_small_ball(labels, img_w, img_h, area_thresh=0.01):
        for label in labels:
            cls, xc, yc, w, h = label
            if cls == 1:  # balle
                if (w * h) < area_thresh:
                    return True
        return False


