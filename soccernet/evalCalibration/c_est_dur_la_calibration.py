
from filterpy.kalman import KalmanFilter
import numpy as np
from CameraFeatureExtractor import CameraFeatureExtractor
from sn_calibration_baseline.camera import Camera
from collections import defaultdict
import copy

def find_neighbor_frame(direction, idx, preds, image_ids, outlier_set, df_conf, alpha, max_lookahead, strategy):
    step = -1 if direction == "before" else 1

    if strategy == "default":
        for i in range(idx + step, 0 if step == -1 else len(image_ids), step):
            fid = image_ids[i]
            if fid not in outlier_set:
                return preds[fid]["camera"]
        return None

    elif strategy == "confidence":
        if df_conf is None:
            raise ValueError("df_conf doit être fourni pour la stratégie 'confidence'")

        best_score = -np.inf
        best_cam = None
        df_conf = df_conf.set_index("image_id")

        for offset in range(1, max_lookahead + 1):
            i = idx + step * offset
            if i < 0 or i >= len(image_ids):
                break

            fid = image_ids[i]
            if fid in outlier_set or fid not in df_conf.index:
                continue

            conf = df_conf.loc[fid, "confidence_score"]
            score = conf - offset * alpha
            if score > best_score:
                best_score = score
                best_cam = preds[fid]["camera"]

        return best_cam

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")


def add_confidence_score(df, model_path):
    import joblib

    model = joblib.load(model_path)
    model_features = model.get_booster().feature_names

    df_renamed = df.copy()

    # Construire une map entre noms sans point (df actuel) et noms attendus par le modèle
    rename_map = {}
    for col in df.columns:
        for model_col in model_features:
            if col.replace("_", "").replace(".", "") == model_col.replace("_", "").replace(".", ""):
                rename_map[col] = model_col
                break

    # Appliquer le renommage pour matcher le modèle
    df_renamed = df_renamed.rename(columns=rename_map)

    # Vérifie si toutes les features nécessaires sont là
    missing = [feat for feat in model_features if feat not in df_renamed.columns]
    if missing:
        raise ValueError(f"❌ Features manquantes dans le DataFrame (même après renommage) : {missing}")

    # Prédiction
    X = df_renamed[model_features]
    y_proba = model.predict_proba(X)[:, 1]

    # Ajouter à la version originale
    df_out = df.copy()
    df_out["confidence_score"] = y_proba

    return df_out


def kalman_1d(values, R=0.5, Q=0.01):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[values[0]], [0.]])  # [position, vitesse]
    kf.F = np.array([[1., 1.], [0., 1.]])  # modèle de mouvement constant
    kf.H = np.array([[1., 0.]])           # on observe uniquement la position
    kf.P *= 1000.
    kf.R = R  # Bruit de mesure
    kf.Q = np.eye(2) * Q  # Bruit de processus

    smoothed = []
    for z in values:
        kf.predict()
        kf.update(z)
        smoothed.append(kf.x[0, 0])
    return smoothed


def apply_kalman_to_camera_preds(preds):
    preds_smoothed = copy.deepcopy(preds)

    # Regrouper les image_id par clip
    clip_dict = defaultdict(list)
    for image_id in preds:
        clip_id = str(image_id).zfill(10)[1:4]  # format CCC
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

        smoothed_params = {key: kalman_1d(values) for key, values in param_sequences.items()}

        for i, image_id in enumerate(image_ids_sorted):
            cam = preds_smoothed[image_id]["camera"]
            cam["pan_degrees"] = smoothed_params["pan_degrees"][i]
            cam["tilt_degrees"] = smoothed_params["tilt_degrees"][i]
            cam["roll_degrees"] = smoothed_params["roll_degrees"][i]
            cam["position_meters"][0] = smoothed_params["pos_x"][i]
            cam["position_meters"][1] = smoothed_params["pos_y"][i]
            cam["position_meters"][2] = smoothed_params["pos_z"][i]

    print("✅ Kalman filter appliqué sur tous les clips.")
    return preds_smoothed


def recalculate_projection_error_on_subset(df, subset_image_ids, preds, gt):
    from sn_calibration_baseline.camera import Camera

    old_errors = []
    new_errors = []
    image_ids_used = []

    subset_image_ids = [str(i) for i in subset_image_ids]
    ex = subset_image_ids[0]

    for image_id in subset_image_ids:
        if image_id not in preds or image_id not in gt:
            print(image_id)
            continue

        cam = preds[image_id]["camera"]
        try:
            cam_obj = Camera(1920, 1080)
            cam_obj.from_json_parameters(cam)
        except Exception as e:
            print(f"❌ Erreur dans from_json_parameters pour {image_id} → {e}")
            continue

        total_err = 0
        count = 0
        for bbox in gt[image_id]["bboxes"]:
            try:
                x_gt, y_gt = bbox["pitch"]
                pt = np.array([*bbox["image"][:2], 1])
                x_proj, y_proj, _ = cam_obj.unproject_point_on_planeZ0(pt)
                total_err += np.linalg.norm([x_proj - x_gt, y_proj - y_gt])
                count += 1

            except:
                print("c'est ici que ça merde?")
                continue

        if count > 0:
            #print(image_id)
            new_err = total_err / count

            old_err_row = df.loc[df["image_id"] == image_id, "reprojection_error"]

            if not old_err_row.empty:
                old_err = old_err_row.iloc[0]
                old_errors.append(old_err)
                new_errors.append(new_err)
                image_ids_used.append(image_id)
                df.loc[df["image_id"] == image_id, "reprojection_error"] = new_err
        else:
            print(f"❌ Frame ignorée → {image_id} (aucune bbox valide / unprojection impossible)")
            continue


    # Mini rapport d’impact
    if len(old_errors) == 0:
        print("❌ Aucun recalcul effectué (aucune frame valide).")
        return

    old_errors = np.array(old_errors)
    new_errors = np.array(new_errors)
    delta = new_errors - old_errors

    # Stats principales
    mean_before = np.mean(old_errors)
    mean_after = np.mean(new_errors)
    median_before = np.median(old_errors)
    median_after = np.median(new_errors)

    print(f"\n📈 Reprojection error recalculée pour {len(old_errors)} frames outliers :")
    print(f"   ➤ Moyenne AVANT  : {mean_before:.2f}")
    print(f"   ➤ Moyenne APRÈS  : {mean_after:.2f}")
    print(f"   ➤ Médiane AVANT  : {median_before:.2f}")
    print(f"   ➤ Médiane APRÈS  : {median_after:.2f}")
    print(f"   ➤ Amélioration moyenne : {np.mean(-delta):.2f}")
    print(f"   ➤ {np.sum(delta < 0)} frames ont été améliorées, {np.sum(delta > 0)} dégradées")

    # 🌡️ Évolution médiane
    evolution = median_before - median_after
    signe = "↘️" if evolution > 0 else ("↗️" if evolution < 0 else "➡️")
    print(f"\n   📊 Évolution de la médiane : {signe} {abs(evolution):.2f} px")


def detect_and_fix_outliers(df, preds, model_path,nan_ids):
    import joblib

    model = joblib.load(model_path)

    expected_features = [
        'pan', 'tilt', 'roll', 'pos_x', 'pos_y', 'pos_z', 'pos_norm',
        'd_pan', 'd_tilt', 'd_roll', 'd_pos_x', 'd_pos_y', 'd_pos_z', 'd_pos_norm',
        'dd_pan', 'dd_tilt', 'dd_roll', 'dd_pos_x', 'dd_pos_y', 'dd_pos_z', 'dd_pos_norm'
    ]
    X = df[expected_features]
    image_ids = df["image_id"].tolist()

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > 0.2).astype(int)

    outliers = [image_id for image_id, pred in zip(image_ids, y_pred) if pred == 1]
    nan_outliers_ids = list(set(outliers + nan_ids))
    preds_fixed = interpolate_outliers(preds, nan_outliers_ids)

    return outliers, preds_fixed


def interpolate_outliers(preds, outliers, df_conf=None, alpha=0.02, max_lookahead=15, strategy="default"):
    image_ids = sorted(preds.keys(), key=lambda x: int(x))
    outlier_set = set(outliers)
    import copy
    preds_interp = copy.deepcopy(preds)
    nb_frame_modified = 0
    for outlier_id in outliers:

        idx = image_ids.index(outlier_id)
        before = find_neighbor_frame("before", idx, preds, image_ids, outlier_set, df_conf, alpha, max_lookahead,
                                     strategy)
        after = find_neighbor_frame("after", idx, preds, image_ids, outlier_set, df_conf, alpha, max_lookahead,
                                    strategy)

        """

        # Chercher frame valide avant
        before = None
        for i in range(idx - 1, -1, -1):
            if image_ids[i] not in outlier_set:
                before = preds[image_ids[i]]["camera"]
                break

        # Chercher frame valide après
        after = None
        for i in range(idx + 1, len(image_ids)):
            if image_ids[i] not in outlier_set:
                after = preds[image_ids[i]]["camera"]
                break
        """
        if before and after:
            interp_cam = {}
            for key in ["pan_degrees", "tilt_degrees", "roll_degrees"]:
                interp_cam[key] = (before[key] + after[key]) / 2

            interp_cam["position_meters"] = [
                (before["position_meters"][i] + after["position_meters"][i]) / 2
                for i in range(3)
            ]




            # Compléter les autres paramètres de calibration à partir de before
            for key in [
                "x_focal_length", "y_focal_length"
            ]:
                interp_cam[key] = (before[key] + after[key]) / 2

            for key in [
                "principal_point",
                "tangential_distortion"
            ]:
                interp_cam[key] = [
                    (before[key][i] + after[key][i]) / 2 for i in range(len(before[key]))
                ]

            for key in [
                "radial_distortion",
                "thin_prism_distortion"
            ]:
                interp_cam[key] = [
                    (before[key][i] + after[key][i]) / 2 for i in range(len(before[key]))
                ]

            preds_interp[outlier_id]["camera"] = interp_cam
            if abs(interp_cam["tilt_degrees"]) < 1 or interp_cam["tilt_degrees"] > 89:
                print(f"⚠️ Tilt extrême interpolé pour {outlier_id}: {interp_cam['tilt_degrees']}°")

            if abs(interp_cam["position_meters"][2]) < 1e-2:
                print(f"⚠️ Z proche de 0 pour {outlier_id}: {interp_cam['position_meters'][2]}")

            nb_frame_modified += 1

    print("Nombre de frame modifié grâce à l'interpolation" ,nb_frame_modified)

    return preds_interp


def print_projection_stats(df, label, preds=None, gt=None, subset_image_ids=None):
    if subset_image_ids:
        errors = []
        for image_id in subset_image_ids:
            if image_id not in preds or image_id not in gt:
                continue
            cam = preds[image_id]["camera"]
            try:
                cam_obj = Camera(1920, 1080)
                cam_obj.from_json_parameters(cam)
            except:
                continue

            total_err = 0
            count = 0
            for bbox in gt[image_id]["bboxes"]:
                try:
                    x_gt, y_gt = bbox["pitch"]
                    pt = np.array([*bbox["image"][:2], 1])
                    x_proj, y_proj, _ = cam_obj.unproject_point_on_planeZ0(pt)
                    total_err += np.linalg.norm([x_proj - x_gt, y_proj - y_gt])
                    count += 1
                except:
                    continue

            if count > 0:
                errors.append(total_err / count)

        errors = np.array(errors)


    else:
        errors = df["reprojection_error"].dropna().to_numpy()

    # Stats classiques
    print(f"\n📊 Statistiques ({label}):")
    if len(errors) == 0:
        print("Aucune donnée disponible.")
        return

    print(f"Moyenne: {np.mean(errors):.2f}")
    print(f"Médiane: {np.median(errors):.2f}")
    print(f"Q1: {np.percentile(errors, 25):.2f}")
    print(f"Q3: {np.percentile(errors, 75):.2f}")
    print(f"Min: {np.min(errors):.2f} / Max: {np.max(errors):.2f}")
    print(f"Écart-type: {np.std(errors):.2f}")
    print(f"Variance: {np.var(errors):.2f}")


if __name__ == "__main__":
    pred_path = "/Users/jeangrifnee/PycharmProjects/soccernet/outputs/sn-gamestate/2025-03-25/Valid_Calib/calib_pred.json"
    gt_path = "/Users/jeangrifnee/PycharmProjects/soccernet/evalCalibration/gt_valid_calibration.json"
    model_path = "calib_clf.pkl"
    extractor = CameraFeatureExtractor(pred_path, gt_path, split="valid")
    df = extractor.build()
    preds_raw = extractor.preds
    # Détection des frames NaN
    nan_ids = []
    for image_id, data in preds_raw.items():
        cam = data.get("camera")
        if cam is None:
            nan_ids.append(image_id)
            continue
        values = [
            cam.get("pan_degrees"),
            cam.get("tilt_degrees"),
            cam.get("roll_degrees"),
            *cam.get("position_meters", [])
        ]
        if not all(np.isfinite(v) for v in values):
            nan_ids.append(image_id)

    print("Nb frame avec Nan_id :", len(nan_ids))
    gt = extractor.gt
    print("On va test le confidence model ")
    model_path_confidence = "full_features_best_model_anchor.pkl"
    test = add_confidence_score(df,model_path_confidence)
    print("Alors voilà ce que ça donne , ", test.head(5))

    print_projection_stats(df, "Avant correction")
    outliers, preds_fixed = detect_and_fix_outliers(df, preds_raw, model_path, nan_ids)
    print("Nb frame avec outliers :", len(outliers))
    all_outliers = list(set(outliers + nan_ids))
    print("Nb frame avecles deux :", len(all_outliers))




    recalculate_projection_error_on_subset(df, outliers, preds_fixed, gt)
    print_projection_stats(df, "Après Interpolation")

    preds_smoothed = apply_kalman_to_camera_preds(preds_fixed)
    all_ids = df["image_id"].tolist()
    recalculate_projection_error_on_subset(df, all_ids, preds_smoothed, gt)
    print_projection_stats(df, "Après Interpolation et Kalman")



    # Stat global après update partiel
   # print_projection_stats(df, "Après correction (sur tout le dataset)")
