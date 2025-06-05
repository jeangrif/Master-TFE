import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from sn_calibration_baseline.camera import Camera


def resample_line(points, n=10):
    if len(points) < 2:
        return points
    x = [p["x"] for p in points]
    y = [p["y"] for p in points]
    try:
        tck, _ = splprep([x, y], s=0)
        unew = np.linspace(0, 1.0, n)
        out = splev(unew, tck)
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(out[0], out[1])]
    except Exception:
        return points


def point_dist(p1, p2):
    return np.linalg.norm([p1["x"] - p2["x"], p1["y"] - p2["y"]])


def evaluate_lines(pred_path, gt_path, n_resample=10):
    with open(pred_path) as f:
        pred_data = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    total_lines = 0
    matched_lines = 0
    total_point_error = 0
    total_points = 0
    missing_lines = 0
    per_frame_stats = []

    print("📉 Nb images dans GT :", len(gt_data))
    print("📦 Nb images dans Pred :", len(pred_data))
    print("❌ Images absentes du Pred :", len(set(gt_data) - set(pred_data)))

    for image_id in tqdm(gt_data.keys(), desc="🔍 Évaluation des lignes"):
        if image_id not in pred_data:
            continue
        gt_lines = gt_data[image_id]
        pred_lines = pred_data[image_id]
        nb_gt_lines = len(gt_lines)
        nb_detected_lines = 0

        for line_name, gt_pts in gt_lines.items():
            total_lines += 1
            if line_name not in pred_lines:
                missing_lines += 1
                continue
            pred_pts = pred_lines[line_name]
            gt_rs = resample_line(gt_pts, n=n_resample)
            pred_rs = resample_line(pred_pts, n=n_resample)
            if len(gt_rs) != len(pred_rs):
                continue
            distances = [point_dist(p1, p2) for p1, p2 in zip(gt_rs, pred_rs)]
            total_point_error += sum(distances)
            total_points += len(distances)
            matched_lines += 1
            nb_detected_lines += 1

        per_frame_stats.append({
            "image_id": image_id,
            "nb_gt_lines": nb_gt_lines,
            "nb_detected_lines": nb_detected_lines,
            "detection_ratio": nb_detected_lines / nb_gt_lines if nb_gt_lines > 0 else 0
        })

    avg_error = total_point_error / total_points if total_points else float("inf")
    recall = matched_lines / total_lines if total_lines else 0
    frame_df = pd.DataFrame(per_frame_stats)
    mean_detection_ratio = frame_df["detection_ratio"].mean()

    print("\n📊 Résultats d'évaluation des lignes :")
    print(f"🔢 Lignes annotées totales      : {total_lines}")
    print(f"✅ Lignes bien détectées        : {matched_lines}")
    print(f"❌ Lignes manquantes            : {missing_lines}")
    print(f"📈 Taux de détection (recall)   : {recall*100:.2f}%")
    print(f"📏 Erreur moyenne par point     : {avg_error:.5f} (coord. normalisées)")
    print(f"\n📈 Pourcentage moyen de lignes détectées par frame : {mean_detection_ratio * 100:.2f}%")

    frame_df.to_csv("per_frame_detection.csv", index=False)
    print("🗂️ Détails par image exportés dans per_frame_detection.csv")


def evaluate_camera_projection(pred_path, gt_path, image_width=1920, image_height=1080):
    with open(pred_path) as f:
        pred_data = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    total_error = 0
    total_points = 0
    all_errors = []
    per_frame_stats = []

    print("\n📸 Évaluation de la calibration (projection bbox_image ➝ pitch)")
    print(f"📂 Frames dans GT : {len(gt_data)}")
    print(f"📂 Frames dans Pred : {len(pred_data)}")

    missing_in_pred = []
    missing_cam_params = []

    for image_id in tqdm(gt_data.keys(), desc="🎯 Projection vers terrain"):
        if image_id not in pred_data:
            missing_in_pred.append(image_id)
            continue

        cam_json = pred_data[image_id].get("camera", None)
        if cam_json is None:
            missing_cam_params.append(image_id)
            continue

        sn_cam = Camera(iwidth=image_width, iheight=image_height)
        sn_cam.from_json_parameters(cam_json)

        bboxes = gt_data[image_id]["bboxes"]
        n_valid = 0
        frame_errors = []

        for bbox in bboxes:
            x, y, w, h = bbox["image"]
            pt = np.array([x, y, 1])
            x_gt, y_gt = bbox["pitch"]
            try:
                x_proj, y_proj, _ = sn_cam.unproject_point_on_planeZ0(pt)
                err = np.linalg.norm([x_proj - x_gt, y_proj - y_gt])
                total_error += err
                total_points += 1
                frame_errors.append(err)
                all_errors.append({
                    "image_id": image_id,
                    "gt": (x_gt, y_gt),
                    "proj": (x_proj, y_proj),
                    "error": err
                })
                n_valid += 1
            except Exception:
                continue

        if n_valid > 0:
            per_frame_stats.append({
                "image_id": image_id,
                "num_bboxes": len(bboxes),
                "valid_bboxes": n_valid,
                "avg_error": np.mean(frame_errors)
            })

    print("\n📉 Frames manquantes dans Pred :", len(missing_in_pred))
    print("📉 Frames sans caméra prédite   :", len(missing_cam_params))
    print("📈 Frames utilisées pour l'évaluation :", len(per_frame_stats))

    valid_errors = [e["error"] for e in all_errors if not np.isnan(e["error"])]
    avg_error = np.mean(valid_errors) if valid_errors else float("inf")
    print(f"\n📏 Erreur moyenne globale de projection terrain : {avg_error:.4f} (unités terrain)")

    df_errors = pd.DataFrame(all_errors)
    df_errors.to_csv("projection_detailed_momenttum_train.csv", index=False)
    print("🗂️ Fichier exporté : projection_detailed.csv")

    df_frames = pd.DataFrame(per_frame_stats)
    df_frames.to_csv("projection_per_frame_momenttum_train.csv", index=False)
    print("🗂️ Fichier exporté : projection_per_frame.csv")

    return avg_error



# === 🎬 EXÉCUTION COMBINÉE ===

if __name__ == "__main__":
    pred_path = "/Users/jeangrifnee/PycharmProjects/soccernet/evalCalibration/predictions_corrected.json"
    #gt_line_path = "gt_test_lines.json"
    gt_calib_path = "gt_train_calibration.json"

    #evaluate_lines(pred_path, gt_line_path)
    evaluate_camera_projection(pred_path, gt_calib_path)
