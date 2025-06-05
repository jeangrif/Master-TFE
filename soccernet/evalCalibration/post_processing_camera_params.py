import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import numpy as np

def apply_momentum_correction_on_preds(
    preds,
    clip_id=None,
    **momentum_kwargs
):
    """
    Applique les corrections de caméra sur les frames détectées par la méthode momentum.
    Retourne un nouveau dictionnaire et la liste des frames corrigées.

    Args:
        preds (dict): JSON déjà chargé
        clip_id (int): clip concerné ou None
        momentum_kwargs: tous les paramètres de ton détecteur robust_momentum

    Returns:
        Tuple[dict, List[str]]: (predictions corrigées, frames modifiées)
    """
    import numpy as np
    from copy import deepcopy

    corrected = deepcopy(preds)

    detected = detect_camera_anomalies_robust_momentum_from_dict(preds, clip_id=clip_id, **momentum_kwargs)
    def extract_clip_id_from_image_id(image_id):
        id_str = str(image_id).zfill(10)
        return int(id_str[1:4])

    previous_params = {}
    momentum = {}
    updated_frames = []

    sorted_keys = sorted(preds.keys(), key=lambda k: int(k))
    for image_id in sorted_keys:
        if clip_id is not None and extract_clip_id_from_image_id(image_id) != clip_id:
            continue

        cam = preds[image_id]["camera"]
        pan = cam.get("pan_degrees", 0)
        tilt = cam.get("tilt_degrees", 0)
        roll = cam.get("roll_degrees", 0)
        pos = np.array(cam.get("position_meters", [0, 0, 0]))

        if not previous_params:
            previous_params = {"pan": pan, "tilt": tilt, "roll": roll, "pos": pos}
            momentum = {"pan": 0, "tilt": 0, "roll": 0, "pos": np.zeros(3)}
            continue

        delta = {
            "pan": pan - previous_params["pan"],
            "tilt": tilt - previous_params["tilt"],
            "roll": roll - previous_params["roll"],
            "pos": pos - previous_params["pos"]
        }

        for k in ["pan", "tilt", "roll"]:
            momentum[k] = momentum_kwargs.get("beta", 0.9) * momentum[k] + (1 - momentum_kwargs.get("beta", 0.9)) * delta[k]
        momentum["pos"] = momentum_kwargs.get("beta", 0.9) * momentum["pos"] + (1 - momentum_kwargs.get("beta", 0.9)) * delta["pos"]

        if image_id in detected:
            corrected[image_id]["camera"]["pan_degrees"] = previous_params["pan"] + momentum["pan"]
            corrected[image_id]["camera"]["tilt_degrees"] = previous_params["tilt"] + momentum["tilt"]
            corrected[image_id]["camera"]["roll_degrees"] = previous_params["roll"] + momentum["roll"]
            corrected[image_id]["camera"]["position_meters"] = (previous_params["pos"] + momentum["pos"]).tolist()
            updated_frames.append(image_id)

        previous_params = {
            "pan": pan,
            "tilt": tilt,
            "roll": roll,
            "pos": pos
        }

    return corrected, updated_frames

def detect_camera_anomalies_robust_momentum_from_dict(
    preds,
    clip_id=None,
    beta=0.9,
    abs_thresholds=None,
    ratio_threshold=3.0,
    warmup=10,
    max_streak=2,
    attenuation=0.1
):
    """
    Variante de ton détecteur, mais sur un dict JSON en RAM.
    Retourne la liste des frames détectées.
    """
    import numpy as np
    from collections import defaultdict

    def extract_clip_id_from_image_id(image_id):
        id_str = str(image_id).zfill(10)
        return int(id_str[1:4])

    if abs_thresholds is None:
        abs_thresholds = {
            "pan": 8.0,
            "tilt": 6.0,
            "roll": 3.0,
            "pos": 2.5
        }

    clips = defaultdict(list)
    for image_id_str, data in preds.items():
        cid = extract_clip_id_from_image_id(image_id_str)
        if clip_id is None or cid == clip_id:
            clips[cid].append((int(image_id_str), data))

    problematic_frames = []

    for cid, frames in clips.items():
        frames.sort(key=lambda x: x[0])
        m = {"pan": 0, "tilt": 0, "roll": 0, "pos": 0}
        t = 0
        streak = 0
        previous = None

        for image_id_int, data in frames:
            image_id_str = str(image_id_int).zfill(10)
            cam = data.get("camera", {})
            pan = cam.get("pan_degrees", 0)
            tilt = cam.get("tilt_degrees", 0)
            roll = cam.get("roll_degrees", 0)
            pos = np.array(cam.get("position_meters", [0, 0, 0]))

            if previous is not None:
                deltas = {
                    "pan": abs(pan - previous["pan"]),
                    "tilt": abs(tilt - previous["tilt"]),
                    "roll": abs(roll - previous["roll"]),
                    "pos": np.linalg.norm(pos - previous["pos"]),
                }

                t += 1
                if t <= warmup:
                    for k in deltas:
                        m[k] = deltas[k]
                    previous = {"pan": pan, "tilt": tilt, "roll": roll, "pos": pos}
                    continue

                alert = False
                for k in deltas:
                    if deltas[k] > max(abs_thresholds[k], m[k] * ratio_threshold):
                        alert = True

                if alert:
                    streak += 1
                else:
                    streak = 0

                if streak > max_streak:
                    problematic_frames.append(image_id_str)
                    for k in deltas:
                        m[k] = beta * m[k] + (1 - beta) * deltas[k] * attenuation
                else:
                    for k in deltas:
                        m[k] = beta * m[k] + (1 - beta) * deltas[k]

            previous = {"pan": pan, "tilt": tilt, "roll": roll, "pos": pos}

    return problematic_frames



def extract_clip_id_from_image_id(image_id):
    id_str = str(image_id).zfill(10)
    return int(id_str[1:4])

def detect_abrupt_camera_changes(json_path, clip_id=None, thresholds=None):
    """
    Détecte les changements brusques de paramètres caméra entre frames consécutives dans un même clip.
    Si clip_id est None, traite tous les clips indépendamment.

    Returns:
        List[str]: liste des image_ids suspectes
    """
    with open(json_path, "r") as f:
        preds = json.load(f)

    if thresholds is None:
        thresholds = {
            "pan": 8.0,
            "tilt": 6.0,
            "roll": 3.0,
            "position": 2.5
        }

    # Organiser les frames par clip_id
    clips = defaultdict(list)
    for image_id_str, data in preds.items():
        cid = extract_clip_id_from_image_id(image_id_str)
        if clip_id is None or cid == clip_id:
            clips[cid].append((int(image_id_str), data))

    problematic_frames = []

    for cid, frames in clips.items():
        # Trier les frames dans chaque clip
        frames.sort(key=lambda x: x[0])
        previous = None

        for image_id_int, data in frames:
            image_id_str = str(image_id_int).zfill(10)
            cam = data.get("camera", {})
            pan = cam.get("pan_degrees", 0)
            tilt = cam.get("tilt_degrees", 0)
            roll = cam.get("roll_degrees", 0)
            pos = np.array(cam.get("position_meters", [0, 0, 0]))

            if previous:
                delta_pan = abs(pan - previous["pan"])
                delta_tilt = abs(tilt - previous["tilt"])
                delta_roll = abs(roll - previous["roll"])
                delta_pos = np.linalg.norm(pos - previous["pos"])

                if (delta_pan > thresholds["pan"] or
                    delta_tilt > thresholds["tilt"] or
                    delta_roll > thresholds["roll"] or
                    delta_pos > thresholds["position"]):
                    problematic_frames.append(image_id_str)

            previous = {
                "pan": pan,
                "tilt": tilt,
                "roll": roll,
                "pos": pos
            }

    return problematic_frames
def extract_clip_id(image_id):
    id_str = str(image_id).zfill(10)
    return int(id_str[1:4])  # 3 chiffres après le premier (set_id)

def extract_frame_idx(image_id):
    id_str = str(image_id).zfill(10)
    return int(id_str[-3:])

def plot_calibration_error_per_frame(csv_path, clip_id, show=True):
    df = pd.read_csv(csv_path)
    df["clip_id"] = df["image_id"].apply(extract_clip_id)
    df["frame_idx"] = df["image_id"].apply(extract_frame_idx)

    df_clip = df[df["clip_id"] == clip_id].sort_values("frame_idx")

    plt.figure(figsize=(10, 5))
    plt.plot(df_clip["frame_idx"], df_clip["avg_error"], marker="o")
    plt.title(f"Évolution de l'erreur de calibration - Clip {clip_id}")
    plt.xlabel("Index de frame")
    plt.ylabel("Erreur moyenne (mètres)")
    plt.grid(True)
    plt.tight_layout()

    if show:
        plt.show()

# Exemple d'appel
if __name__ == "__main__":
    csv_path = "projection_per_frame.csv"
    plot_calibration_error_per_frame(csv_path, clip_id=100)
    json_path = "/Users/jeangrifnee/PycharmProjects/soccernet/outputs/sn-gamestate/2025-03-25/Train_Calib/calib_pred.json"
    #abrupt_changes = detect_abrupt_camera_changes(json_path, clip_id=100)
    #print("Nb frames détectées : ", len(abrupt_changes))
    #print("Frames avec changements brusques détectés :", abrupt_changes)
    with open(json_path, "r") as f:
        preds = json.load(f)

    momentum_suspects = detect_camera_anomalies_robust_momentum_from_dict(preds, clip_id=100)
    print("Frames détectées (momentum) :", len(momentum_suspects))
    print(momentum_suspects)
    """
    corrected_preds, corrected_frames = apply_momentum_correction_on_preds(
        preds,
        clip_id=100,
        beta=0.9,
        ratio_threshold=3.0,
        warmup=10,
        max_streak=2,
        attenuation=0.1
    )

    print("Frames corrigées :", len(corrected_frames))

    with open("predictions_corrected.json", "w") as f:
        json.dump(corrected_preds, f, indent=2)
    """