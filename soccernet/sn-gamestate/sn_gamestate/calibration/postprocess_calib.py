import logging
from tracklab.pipeline import ImageLevelModule
from tracklab.utils.collate import default_collate, Unbatchable
import joblib
import torch
import pandas as pd
from typing import Any
import copy
from collections import defaultdict
from filterpy.kalman import KalmanFilter
import numpy as np
from sn_calibration_baseline.camera import Camera
import json
log = logging.getLogger(__name__)


class CalibrationRefiner(ImageLevelModule):
    input_columns = []
    output_columns = ["camera"]
    collate_fn = default_collate

    def __init__(self , image_width, image_height,outlier_model_path, batch_size=750, **kwargs):
        print("✅ CalibrationRefiner initialisé")
        super().__init__(batch_size=batch_size)
        self.image_width = image_width
        self.image_height = image_height
        self.outlier_model = joblib.load(outlier_model_path)
        log.info("✅ CalibrationRefiner loaded.")

    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return {}

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):

        detections = detections.copy()

        # === Étape 0 : Conversion des paramètres caméra + features
        df_cam, preds = self._extract_camera_features(metadatas)

        # === Étape 1 : Prédiction des outliers
        outlier_ids = self._detect_outliers(df_cam)

        # === Étape 2 : Détection NaN ou invalides
        nan_ids = self._detect_invalid_cameras(preds)

        # === Étape 3 : Interpolation des caméras aberrantes
        all_outliers = list(set(outlier_ids + nan_ids))
        preds_fixed = self.interpolate_outliers(preds, all_outliers)

        # === Étape 4 : Lissage Kalman
        preds_smoothed = self.apply_kalman_to_camera_preds(preds_fixed)

        # === Étape 5 : Réinjection dans detections
        if "camera" not in detections.columns:
            detections["camera"] = None
        detections["camera_before"] = detections["camera"].copy()

        for image_id in detections["image_id"]:
            if image_id in preds_smoothed:
                detections.loc[detections["image_id"] == image_id, "camera"] = preds_smoothed[image_id]["camera"]

        log.info(f"📌 Caméras mises à jour dans detections : {len(preds_smoothed)}")

        # === Étape 6 : Recalcul global des bbox_pitch avec caméras corrigées
        bbox_pitch_recomputed = []
        for i, row in detections.iterrows():
            image_id = row["image_id"]
            bbox = row.get("bbox_ltwh")
            if bbox is not None and isinstance(bbox, (list, tuple, np.ndarray)) and image_id in preds_smoothed:

                ltwh = row["bbox_ltwh"]
                ltrb = [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]]

                cam_params = preds_smoothed[image_id]["camera"]
                cam = Camera(iwidth=self.image_width, iheight=self.image_height)
                cam.from_json_parameters(cam_params)
                bbox_pitch_func = get_bbox_pitch(cam)

                pitch = bbox_pitch_func(ltrb)
                bbox_pitch_recomputed.append(pitch)
            else:
                bbox_pitch_recomputed.append(None)

        detections["bbox_pitch"] = bbox_pitch_recomputed

        return detections

    def interpolate_outliers(self, preds, outliers):
        image_ids = sorted(preds.keys(), key=lambda x: int(x))
        outlier_set = set(outliers)
        preds_interp = copy.deepcopy(preds)

        for outlier_id in outliers:
            idx = image_ids.index(outlier_id)

            # Frame avant
            before = None
            for i in range(idx - 1, -1, -1):
                fid = image_ids[i]
                if fid not in outlier_set:
                    before = preds[fid]["camera"]
                    break

            # Frame après
            after = None
            for i in range(idx + 1, len(image_ids)):
                fid = image_ids[i]
                if fid not in outlier_set:
                    after = preds[fid]["camera"]
                    break

            if before and after:
                interp_cam = {}
                for key in ["pan_degrees", "tilt_degrees", "roll_degrees"]:
                    interp_cam[key] = (before[key] + after[key]) / 2
                interp_cam["position_meters"] = [(before["position_meters"][i] + after["position_meters"][i]) / 2 for i in range(3)]

                for key in [
                    "x_focal_length", "y_focal_length", "principal_point",
                    "tangential_distortion", "radial_distortion", "thin_prism_distortion"
                ]:
                    if isinstance(before[key], list):
                        interp_cam[key] = [(before[key][i] + after[key][i]) / 2 for i in range(len(before[key]))]
                    else:
                        interp_cam[key] = (before[key] + after[key]) / 2

                preds_interp[outlier_id]["camera"] = interp_cam

        return preds_interp

    def apply_kalman_to_camera_preds(self, preds):
        preds_smoothed = copy.deepcopy(preds)
        clip_dict = defaultdict(list)
        for image_id in preds:
            clip_id = str(image_id).zfill(10)[1:4]
            clip_dict[clip_id].append(image_id)

        for clip_id, image_ids in clip_dict.items():
            image_ids_sorted = sorted(image_ids, key=lambda x: int(x))

            param_sequences = {
                "pan_degrees": [],
                "tilt_degrees": [],
                "roll_degrees": [],
                "pos_x": [],
                "pos_y": [],
                "pos_z": [],
            }

            for image_id in image_ids_sorted:
                cam = preds[image_id]["camera"]
                param_sequences["pan_degrees"].append(cam["pan_degrees"])
                param_sequences["tilt_degrees"].append(cam["tilt_degrees"])
                param_sequences["roll_degrees"].append(cam["roll_degrees"])
                param_sequences["pos_x"].append(cam["position_meters"][0])
                param_sequences["pos_y"].append(cam["position_meters"][1])
                param_sequences["pos_z"].append(cam["position_meters"][2])

            smoothed = {k: self.kalman_1d(v) for k, v in param_sequences.items()}

            for i, image_id in enumerate(image_ids_sorted):
                cam = preds_smoothed[image_id]["camera"]
                cam["pan_degrees"] = smoothed["pan_degrees"][i]
                cam["tilt_degrees"] = smoothed["tilt_degrees"][i]
                cam["roll_degrees"] = smoothed["roll_degrees"][i]
                cam["position_meters"][0] = smoothed["pos_x"][i]
                cam["position_meters"][1] = smoothed["pos_y"][i]
                cam["position_meters"][2] = smoothed["pos_z"][i]

        return preds_smoothed

    def kalman_1d(self, values, R=0.5, Q=0.01):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[values[0]], [0.]])
        kf.F = np.array([[1., 1.], [0., 1.]])
        kf.H = np.array([[1., 0.]])
        kf.P *= 1000.
        kf.R = R
        kf.Q = np.eye(2) * Q

        smoothed = []
        for z in values:
            kf.predict()
            kf.update(z)
            smoothed.append(kf.x[0, 0])
        return smoothed

    def _detect_invalid_cameras(self, preds):
        nan_ids = []
        for image_id, data in preds.items():
            cam = data.get("camera")
            if cam is None:
                nan_ids.append(image_id)
                continue
            vals = [
                cam.get("pan_degrees"), cam.get("tilt_degrees"), cam.get("roll_degrees"),
                *cam.get("position_meters", [])
            ]
            if not all(np.isfinite(v) for v in vals):
                nan_ids.append(image_id)

        if "2095000001" in preds:
            log.info(f"🔍 Caméra INITIALE pour 2095000001 : {json.dumps(preds['2095000001']['camera'], indent=2)}")

        return nan_ids

    def _detect_outliers(self, df_cam_renamed):
        X = df_cam_renamed[[
            "pan", "tilt", "roll",
            "pos_x", "pos_y", "pos_z", "pos_norm",
            "d_pan", "d_tilt", "d_roll",
            "d_pos_x", "d_pos_y", "d_pos_z", "d_pos_norm",
            "dd_pan", "dd_tilt", "dd_roll",
            "dd_pos_x", "dd_pos_y", "dd_pos_z", "dd_pos_norm"
        ]]
        y_proba = self.outlier_model.predict_proba(X)[:, 1]
        y_pred = (y_proba > 0.2).astype(int)
        return df_cam_renamed["image_id"][y_pred == 1].tolist()

    def _extract_camera_features(self, metadatas):
        metadatas_reset = metadatas.reset_index(drop=True)
        df_cam = pd.DataFrame(list(metadatas_reset["parameters"]))
        df_cam["image_id"] = metadatas_reset["id"].astype(str)

        df_cam["pos_x"] = df_cam["position_meters"].apply(lambda x: x[0])
        df_cam["pos_y"] = df_cam["position_meters"].apply(lambda x: x[1])
        df_cam["pos_z"] = df_cam["position_meters"].apply(lambda x: x[2])
        df_cam["pos_norm"] = np.linalg.norm(df_cam[["pos_x", "pos_y", "pos_z"]].values, axis=1)

        for name in ["pan_degrees", "tilt_degrees", "roll_degrees", "pos_x", "pos_y", "pos_z", "pos_norm"]:
            df_cam[f"d_{name}"] = df_cam[name].diff().fillna(0)
            df_cam[f"dd_{name}"] = df_cam[f"d_{name}"].diff().fillna(0)

        rename_map = {
            "pan_degrees": "pan", "tilt_degrees": "tilt", "roll_degrees": "roll",
            "d_pan_degrees": "d_pan", "d_tilt_degrees": "d_tilt", "d_roll_degrees": "d_roll",
            "dd_pan_degrees": "dd_pan", "dd_tilt_degrees": "dd_tilt", "dd_roll_degrees": "dd_roll",
        }
        df_cam_renamed = df_cam.rename(columns=rename_map)

        return df_cam_renamed, {str(k): {"camera": v} for k, v in
                                metadatas.set_index("id")["parameters"].to_dict().items()}
def get_bbox_pitch(cam):
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l+(r-l)/2, b, 1])

        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox
