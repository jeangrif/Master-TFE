import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def extract_clip_id(image_id):
    return int(str(image_id).zfill(10)[1:4])

def extract_frame_idx(image_id):
    return int(str(image_id)[-3:])

def get_camera_params(cam):
    return {
        "pan": cam["pan_degrees"],
        "tilt": cam["tilt_degrees"],
        "roll": cam["roll_degrees"],
        "pos_x": cam["position_meters"][0],
        "pos_y": cam["position_meters"][1],
        "pos_z": cam["position_meters"][2],
        "pos_norm": np.linalg.norm(cam["position_meters"]),
    }

def build_feature_dataframe(preds, gt, split):
    all_data = []
    line_types_to_track = set()
    for data in preds.values():
        lines = data.get("lines", {})
        line_types_to_track.update(lines.keys())
    line_types_to_track = sorted(list(line_types_to_track))

    # Grouper les frames par clip
    clips = {}
    for fid in preds:
        clip = extract_clip_id(fid)
        clips.setdefault(clip, []).append(fid)

    for clip_id, fids in tqdm(clips.items(), desc=f"Processing {split}"):
        fids = sorted(fids, key=lambda x: int(x))
        prev = None
        prev_deltas = None

        for i, fid in enumerate(fids):
            if fid not in preds or fid not in gt:
                continue

            cam = preds[fid]["camera"]
            if not all(np.isfinite([
                cam["pan_degrees"], cam["tilt_degrees"], cam["roll_degrees"],
                *cam["position_meters"]
            ])):
                continue  # skip NaNs

            params = get_camera_params(cam)

            # 1ère dérivées
            if prev:
                deltas = {k: params[k] - prev[k] for k in params}
            else:
                deltas = {k: 0 for k in params}

            # 2e dérivées (momentum)
            if prev_deltas:
                momenta = {f"dd_{k}": deltas[k] - prev_deltas[k] for k in deltas}
            else:
                momenta = {f"dd_{k}": 0 for k in deltas}

            prev = params
            prev_deltas = deltas

            # Erreur de reprojection GT
            total_err = 0
            count = 0
            for bbox in gt[fid]["bboxes"]:
                x_gt, y_gt = bbox["pitch"]
                try:
                    pt = np.array([*bbox["image"][:2], 1])
                    cam_obj = Camera(1920, 1080)
                    cam_obj.from_json_parameters(cam)
                    x_proj, y_proj, _ = cam_obj.unproject_point_on_planeZ0(pt)
                    total_err += np.linalg.norm([x_proj - x_gt, y_proj - y_gt])
                    count += 1
                except:
                    continue

            if count == 0:
                print(f"⚠️ Aucune bbox projetée pour frame {fid}")
                continue

            row = {
                "image_id": fid,
                "clip_id": clip_id,
                "frame_idx": extract_frame_idx(fid),
                "split": split,
                "reprojection_error": total_err / count,
            }
            row.update(params)
            row.update({f"d_{k}": v for k, v in deltas.items()})
            row.update(momenta)

            # 📌 AJOUT : features basés sur les lignes détectées
            lines = preds[fid].get("lines", {})
            line_features = {}

            # Statistiques globales
            line_features["nb_lines_detected"] = len(lines)
            line_features["total_points_detected"] = sum(len(pts) for pts in lines.values())

            # Pour chaque type de ligne suivi
            for line_type in line_types_to_track:
                key = line_type.lower().replace(" ", "_")
                points = lines.get(line_type, [])
                line_features[f"has_{key}"] = int(len(points) > 0)
                line_features[f"nb_pts_{key}"] = len(points)

                if len(points) >= 2:
                    distances = [
                        np.linalg.norm([
                            points[i + 1]["x"] - points[i]["x"],
                            points[i + 1]["y"] - points[i]["y"]
                        ]) for i in range(len(points) - 1)
                    ]
                    line_features[f"avg_len_{key}"] = np.mean(distances)
                else:
                    line_features[f"avg_len_{key}"] = 0.0

            row.update(line_features)

            all_data.append(row)

    return pd.DataFrame(all_data)

# -- Main --

if __name__ == "__main__":
    from sn_calibration_baseline.camera import Camera

    root = "/Users/jeangrifnee/PycharmProjects/soccernet/outputs/sn-gamestate/2025-03-25"

    sets = [
        ("Train_Calib", "gt_train_calibration.json", "train"),
        ("Valid_Calib", "gt_valid_calibration.json", "valid")
    ]

    for calib_folder, gt_file, split in sets:
        pred_path = os.path.join(root, calib_folder, "calib_pred.json")
        gt_path = gt_file  # GT accessible directement dans le dossier courant

        with open(pred_path) as f:
            preds = json.load(f)
        with open(gt_path) as f:
            gt = json.load(f)

        df = build_feature_dataframe(preds, gt, split)
        output_path = f"v2features_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Fichier exporté : {output_path}")
