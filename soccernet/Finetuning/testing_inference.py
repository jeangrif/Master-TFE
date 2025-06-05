import os
import cv2
import json
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO
import torch
# =============== PARAMÈTRES ===============
FRAME_FOLDER = "/Users/jeangrifnee/PycharmProjects/soccernet/data/SoccerNetGS/valid/SNGS-095/img1"
PLAYER_MODEL_PATH = "/Users/jeangrifnee/PycharmProjects/soccernet/Finetuning/finetuning_player_detection/fine_tuned/weights/best.pt"
BALL_MODEL_PATH = "/Users/jeangrifnee/PycharmProjects/soccernet/Finetuning/finetuning_ball_detection2/fine_tuned/weights/best.pt"
TILE_SIZE = 384
CLASS_ID_PLAYER = 0
CLASS_ID_BALL = 1
CONF_THRESH = 0.45

# =============== MODELES WRAPPÉS ===============
class YOLOPlayerModel:
    def __init__(self, weights_path, mapped_class_id=CLASS_ID_PLAYER):
        self.model = YOLO(weights_path)
        if torch.backends.mps.is_available():
            self.model.to('mps')  # 💻 GPU Apple Silicon
        self.class_id = mapped_class_id

    def infer_player_only(self, image, conf_thresh=0.3):
        results = self.model(image, verbose=False)[0]
        preds = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            preds.append((self.class_id, conf, x1, y1, x2, y2))
        return preds

class YOLOBallModel:
    def __init__(self, weights_path, mapped_class_id=CLASS_ID_BALL):
        self.model = YOLO(weights_path)
        if torch.backends.mps.is_available():
            self.model.to('mps')  # 💻 GPU Apple Silicon
        self.class_id = mapped_class_id

    def infer_ball_only(self, image, conf_thresh=0.3):
        results = self.model(image, verbose=False)[0]
        preds = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            preds.append((self.class_id, conf, x1, y1, x2, y2))
        return preds

# =============== MOTEUR D’INFÉRENCE ===============
class BallInferenceEngine:
    def __init__(self, player_model, ball_model, tile_size=384, conf_thresh=0.3):
        self.player_model = player_model
        self.ball_model = ball_model
        self.tile_size = tile_size
        self.conf_thresh = conf_thresh
        self.last_ball_bbox = None

    def generate_tile_centered_on_bbox(self, image, bbox):
        h, w = image.shape[:2]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        x0 = max(0, min(w - self.tile_size, cx - self.tile_size // 2))
        y0 = max(0, min(h - self.tile_size, cy - self.tile_size // 2))
        tile = image[y0:y0 + self.tile_size, x0:x0 + self.tile_size]
        return [(tile, (x0, y0))]

    def generate_all_tiles(self, image, overlap_ratio=0.25):
        h, w = image.shape[:2]
        stride = int(self.tile_size * (1 - overlap_ratio))  # ex: 384 * 0.75 = 288
        tiles = []
        for y in range(0, h - self.tile_size + 1, stride):
            for x in range(0, w - self.tile_size + 1, stride):
                tile = image[y:y + self.tile_size, x:x + self.tile_size]
                tiles.append((tile, (x, y)))
        return tiles

    def update(self, image) -> Tuple[List, List]:
        preds_players = self.player_model.infer_player_only(image, self.conf_thresh)

        # Heuristique basique pour tenter de retrouver la balle dans les joueurs (inutile ici, donc on saute)
        # On passe directement aux tiles si nécessaire
        preds_ball = []

        if self.last_ball_bbox:
            tiles = self.generate_tile_centered_on_bbox(image, self.last_ball_bbox)
            for tile, (x_off, y_off) in tiles:
                preds = self.ball_model.infer_ball_only(tile, self.conf_thresh)
                for cls, conf, x1, y1, x2, y2 in preds:
                    global_box = (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)
                    preds_ball.append((cls, conf, *global_box))
                    self.last_ball_bbox = global_box
                    print(f"🎯 Balle détectée (conf={conf:.3f}) à {global_box}")
            if preds_ball:
                return preds_players, preds_ball
            else:
                self.last_ball_bbox = None

        # Fallback total
        tiles = self.generate_all_tiles(image)
        print(f"🔍 Fallback total : {len(tiles)} tiles générés.")
        for tile, (x_off, y_off) in tiles:
            preds = self.ball_model.infer_ball_only(tile, self.conf_thresh)
            for cls, conf, x1, y1, x2, y2 in preds:
                global_box = (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)
                preds_ball.append((cls, conf, *global_box))
                self.last_ball_bbox = global_box
                print(f"🎯 Balle retrouvée via fallback (conf={conf:.3f}) à {global_box}")

        if not preds_ball:
            print("❌ Balle totalement perdue.")
            self.last_ball_bbox = None

        return preds_players, preds_ball

# =============== INFÉRENCE SUR DOSSIER ===============
def run_inference_on_folder():
    player_model = YOLOPlayerModel(PLAYER_MODEL_PATH)
    ball_model = YOLOBallModel(BALL_MODEL_PATH)
    engine = BallInferenceEngine(player_model, ball_model, tile_size=TILE_SIZE, conf_thresh=CONF_THRESH)

    frame_paths = sorted(Path(FRAME_FOLDER).glob("*.jpg"))
    output = {}

    for idx, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"Erreur lecture {frame_path}")
            continue

        preds_players, preds_ball = engine.update(image)
        all_preds = preds_players + preds_ball

        output[frame_path.stem] = [
            {
                "class_id": cls,
                "conf": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            }
            for cls, conf, x1, y1, x2, y2 in all_preds
        ]

        print(f"[{idx + 1}/{len(frame_paths)}] Frame {frame_path.stem}: {len(all_preds)} détections")

    with open("predictions_adaptatives.json", "w") as f:
        json.dump(output, f, indent=2)
    print("✅ Inférence terminée. Résultats dans predictions_adaptatives.json")

    # VIDEO DE DEBUG BALLE
    print("🎥 Génération de la vidéo avec uniquement les détections de balle...")
    output_video_path = "balle_detections.mp4"
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        frame_id = frame_path.stem
        for pred in output.get(frame_id, []):
            if pred["class_id"] == CLASS_ID_BALL:
                x1, y1, x2, y2 = pred["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ball {pred['conf']:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Vidéo enregistrée sous : {output_video_path}")

# =============== MAIN ===============
if __name__ == "__main__":
    run_inference_on_folder()
