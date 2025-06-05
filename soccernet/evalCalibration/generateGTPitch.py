import os
import json
from tqdm import tqdm


def generate_ground_truth_json(
        root_dir,
        subset,
        output_file=None,
        task="lines",  # "lines" ou "calibration"
        max_clips=None,
        max_frames_per_clip=None
):
    assert task in ["lines", "calibration"], "task doit être 'lines' ou 'calibration'"

    SET_MAP = {"train": "1", "valid": "2", "test": "3", "challenge": "2"}
    subset_digit = SET_MAP[subset]
    subset_dir = os.path.join(root_dir, subset)

    if output_file is None:
        output_file = f"gt_{subset}_{task}v1.json"

    gt_dict = {}

    video_dirs = sorted(
        [os.path.join(subset_dir, d) for d in os.listdir(subset_dir)
         if os.path.isdir(os.path.join(subset_dir, d))],
        key=lambda x: int(x.split('-')[-1])
    )

    if max_clips:
        video_dirs = video_dirs[:max_clips]

    for video_dir in tqdm(video_dirs, desc=f"📦 Processing {subset}"):
        video_num = video_dir.split("-")[-1]
        labels_path = os.path.join(video_dir, "Labels-GameState.json")

        if not os.path.exists(labels_path):
            print(f"⚠️ Missing annotation file: {labels_path}")
            continue

        with open(labels_path, "r") as f:
            annotations_json = json.load(f)

        frame_annotations = {}

        for ann in annotations_json["annotations"]:
            full_image_id = ann["image_id"]

            if task == "lines":
                if ann.get("supercategory") != "pitch" or "lines" not in ann:
                    continue

                if full_image_id not in frame_annotations:
                    frame_annotations[full_image_id] = {}

                for line_name, points in ann["lines"].items():
                    frame_annotations[full_image_id][line_name] = points

            elif task == "calibration":
                if ann.get("category_id") not in [1, 2, 3]:
                    continue
                if not isinstance(ann.get("bbox_pitch"), dict) or not isinstance(ann.get("bbox_image"), dict):
                    continue

                pitch = ann["bbox_pitch"]
                image = ann["bbox_image"]

                pitch_coords = pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")
                image_coords = image.get("x_center"), image.get("y"), image.get("w"), image.get("h")

                if None in pitch_coords or None in image_coords:
                    continue

                if full_image_id not in frame_annotations:
                    frame_annotations[full_image_id] = []

                frame_annotations[full_image_id].append({
                    "image": list(image_coords),
                    "pitch": list(pitch_coords)
                })

        sorted_ids = sorted(frame_annotations.keys())
        if max_frames_per_clip:
            sorted_ids = sorted_ids[:max_frames_per_clip]

        for full_image_id in sorted_ids:
            gt_dict[full_image_id] = (
                {"bboxes": frame_annotations[full_image_id]}
                if task == "calibration"
                else frame_annotations[full_image_id]
            )

    with open(output_file, "w") as f:
        json.dump(gt_dict, f, indent=2)

    print(f"✅ Ground truth ({task}) saved to {output_file} with {len(gt_dict)} frames.")


# 🧪 Exemple d'appel
if __name__ == "__main__":
    generate_ground_truth_json(
        root_dir="../data/SoccerNetGS",
        subset="valid",
        task="calibration",  # ou "lines"
        max_clips=None,
        max_frames_per_clip=None
    )
