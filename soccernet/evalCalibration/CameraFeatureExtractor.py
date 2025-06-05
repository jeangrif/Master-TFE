import json
import numpy as np
import pandas as pd
from tqdm import tqdm

class CameraFeatureExtractor:
    def __init__(self, pred_path, gt_path, split="unknown"):
        from sn_calibration_baseline.camera import Camera  # vérifie dispo avant toute chose
        self.Camera = Camera

        with open(pred_path) as f:
            self.preds = {k: v for k, v in json.load(f).items() if k.isdigit()}

        with open(gt_path) as f:
            self.gt = {k: v for k, v in json.load(f).items() if k.isdigit()}

        self.split = split
        self.lines = {k: v.get("lines", {}) for k, v in self.preds.items()}


    @staticmethod
    def extract_clip_id(image_id):
        return int(str(image_id).zfill(10)[1:4])

    @staticmethod
    def extract_frame_idx(image_id):
        return int(str(image_id)[-3:])

    @staticmethod
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

    def build(self):
        all_data = []
        line_types_to_track = set()
        for lines in self.lines.values():
            line_types_to_track.update(lines.keys())
        line_types_to_track = sorted(list(line_types_to_track))
        ici =0
        # Grouper les frames par clip
        clips = {}
        for fid in self.preds:
            clip = self.extract_clip_id(fid)
            clips.setdefault(clip, []).append(fid)

        for clip_id, fids in tqdm(clips.items(), desc=f"Processing {self.split}"):
            fids = sorted(fids, key=lambda x: int(x))
            prev = None
            prev_deltas = None


            for fid in fids:
                if fid not in self.preds or fid not in self.gt:
                    continue

                cam = self.preds[fid]["camera"]
                if not all(np.isfinite([
                    cam["pan_degrees"], cam["tilt_degrees"], cam["roll_degrees"],
                    *cam["position_meters"]
                ])):
                    ici+=1
                    continue  # skip NaNs

                params = self.get_camera_params(cam)

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
                for bbox in self.gt[fid]["bboxes"]:
                    x_gt, y_gt = bbox["pitch"]
                    try:
                        pt = np.array([*bbox["image"][:2], 1])
                        cam_obj = self.Camera(1920, 1080)
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
                    "frame_idx": self.extract_frame_idx(fid),
                    "split": self.split,
                    "reprojection_error": total_err / count,
                }
                row.update(params)
                row.update({f"d_{k}": v for k, v in deltas.items()})
                row.update(momenta)
                # --- Enrichissement avec features des lignes ---
                line_features = {}
                lines = self.lines.get(fid, {})

                # Statistiques globales
                line_features["nb_lines_detected"] = len(lines)
                line_features["total_points_detected"] = sum(len(pts) for pts in lines.values())

                # Pour chaque type de ligne détectée
                for line_type in line_types_to_track:
                    key = line_type.lower().replace(" ", "_").replace(".", "")
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
        print("Est ce que c'est npormal ? ", ici)

        return pd.DataFrame(all_data)

    def debug_info(self):
        preds_keys = set(self.preds.keys())
        gt_keys = set(self.gt.keys())

        print(f"Nombre de frames dans preds : {len(preds_keys)}")
        print(f"Nombre de frames dans gt    : {len(gt_keys)}")
        print(f"Frames en commun            : {len(preds_keys & gt_keys)}")
        print(f"Frames uniquement dans preds: {len(preds_keys - gt_keys)}")
        print(f"Frames uniquement dans gt   : {len(gt_keys - preds_keys)}")
