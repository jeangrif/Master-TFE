import torch
from ultralytics import YOLO
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def debug_image_prediction(img_tensor, result, title="Debug Image"):
    """
    Affiche l'image passée au modèle + les prédictions YOLO (même si vides)
    """

    # 🔁 Conversion : Tensor CHW → numpy HWC
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_title(title)
    ax.axis("off")

    if result.boxes.shape[0] == 0:
        print("❌ Aucune prédiction sur cette image.")
        plt.show()
        return

    preds = result.boxes
    for i in range(preds.shape[0]):
        box = preds.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = box
        cls = int(preds.cls[i].cpu().item())
        conf = float(preds.conf[i].cpu().item())

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2,
                             edgecolor='red' if cls == 0 else 'blue',
                             facecolor='none')
        label = "Player" if cls == 0 else "Ball"
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{label} ({conf:.2f})",
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.show()

class YOLOModel:
    def __init__(self, pretrained_model_path, device="mps"):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(pretrained_model_path)
        self.model.to(self.device)
        print(f"✅ Modèle chargé sur {self.device.upper()}")

    @torch.no_grad()
    def inference(self, dataloader, conf_threshold=0.25):
        self.model.conf = conf_threshold

        all_predictions = []
        total_frames = 0
        start_time = time.time()

        for imgs, _, image_ids, _ in tqdm(dataloader, desc="🔎 Inference en cours", unit="batch"):
            # Convertir chaque tenseur en tableau NumPy
            imgs_np = [
                (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for img in imgs
            ]

            results = self.model.predict(source=imgs_np, device=self.device, verbose=False)

            for img, res, image_id in zip(imgs,results, image_ids):
                preds = res.boxes.cpu().numpy()
                #debug_image_prediction(img, res, title=f"Image: {image_id}")
                all_predictions.append({
                    "image_id": image_id,
                    "boxes": preds.xyxy,
                    "scores": preds.conf,
                    "classes": preds.cls
                })
                total_frames += 1

        end_time = time.time()
        fps = total_frames / (end_time - start_time)
        print(f"🚀 Inference terminée à {fps:.2f} FPS.")

        return all_predictions, fps

    def train(self, data_yaml_path, epochs=50, save_dir="runs/train", lr=0.01, batch=16, img_size=640):
        """
        Fine-tune le modèle YOLOv8
        """
        overrides = dict(
            blur=0.0,
            median=0.0,
            to_gray=0.0
        )
        print("🚧 Démarrage du fine-tuning YOLO...")
        self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            lr0=lr,
            device=self.device,
            project=save_dir,
            name="fine_tuned",
            conf = 0.3,
            exist_ok=True,

        )
        print("✅ Fine-tuning terminé !")

    def evaluate(self, predictions, dataloader, iou_threshold=0.5, conf_threshold=0.25):
        from ultralytics.utils.metrics import bbox_iou
        import torch
        import numpy as np

        classes = [0, 1]  # 0 = Person, 1 = Ball
        class_names = ["Person", "Ball"]
        stats = {c: {"tp": 0, "fp": 0, "fn": 0, "tpfp": []} for c in classes}
        num_frames = 0

        pred_dict = {pred["image_id"]: pred for pred in predictions}

        for batch in dataloader:
            images, gt_labels_batch, image_ids, sizes = batch

            for idx_in_batch in range(len(images)):


                image_id = image_ids[idx_in_batch]

                if image_id not in pred_dict:
                    continue

                pred_sample = pred_dict[image_id]
                labels = gt_labels_batch[idx_in_batch]
                tile_w, tile_h = sizes[idx_in_batch]

                gt_boxes_by_class = {c: [] for c in classes}
                for ann in labels:
                    cls, xc, yc, w, h = ann
                    cls = int(cls)
                    if cls in gt_boxes_by_class:
                        x1 = (xc - w / 2) * tile_w
                        y1 = (yc - h / 2) * tile_h
                        x2 = (xc + w / 2) * tile_w
                        y2 = (yc + h / 2) * tile_h
                        gt_boxes_by_class[cls].append([x1, y1, x2, y2])

                for c in classes:
                    gt_boxes = torch.tensor(gt_boxes_by_class[c], dtype=torch.float32)
                    detected_gt = set()

                    predictions_by_class = [
                        (box, score)
                        for box, score, cls_pred in
                        zip(pred_sample["boxes"], pred_sample["scores"], pred_sample["classes"])
                        if score >= conf_threshold and int(cls_pred) == c
                    ]

                    for box, score in predictions_by_class:
                        if len(gt_boxes) == 0:
                            stats[c]["fp"] += 1
                            stats[c]["tpfp"].append((0, score))
                            continue

                        box_tensor = torch.tensor(box, dtype=torch.float32).unsqueeze(0)
                        ious = bbox_iou(box_tensor, gt_boxes).squeeze(0)

                        best_iou_val, best_iou_idx = torch.max(ious, dim=0)
                        best_iou_val = float(best_iou_val.item())
                        best_iou_idx = best_iou_idx.item()

                        if best_iou_val >= iou_threshold and best_iou_idx not in detected_gt:
                            stats[c]["tp"] += 1
                            stats[c]["tpfp"].append((1, score))
                            detected_gt.add(best_iou_idx)
                        else:
                            stats[c]["fp"] += 1
                            stats[c]["tpfp"].append((0, score))

                    stats[c]["fn"] += len(gt_boxes) - len(detected_gt)

                num_frames += 1

        # Résumé et calculs des métriques
        aps = []
        for c in classes:
            tp = stats[c]["tp"]
            fp = stats[c]["fp"]
            fn = stats[c]["fn"]
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2 * precision * recall / (precision + recall + 1e-16)

            # --- Calcul AP@0.5 ---
            tpfp = stats[c]["tpfp"]
            if len(tpfp) > 0:
                tpfp_sorted = sorted(tpfp, key=lambda x: -x[1])  # tri décroissant par score
                tps = np.array([x[0] for x in tpfp_sorted])
                fps = 1 - tps
                cum_tp = np.cumsum(tps)
                cum_fp = np.cumsum(fps)

                recalls = cum_tp / (tp + fn + 1e-16)
                precisions = cum_tp / (cum_tp + cum_fp + 1e-16)

                # interpolation 11 points style VOC (simple mais robuste)
                ap = 0.0
                for r_thresh in np.linspace(0, 1, 11):
                    prec_at_r = precisions[recalls >= r_thresh]
                    p = max(prec_at_r) if len(prec_at_r) > 0 else 0.0
                    ap += p / 11
            else:
                ap = 0.0

            stats[c]["precision"] = precision
            stats[c]["recall"] = recall
            stats[c]["f1"] = f1
            stats[c]["ap50"] = ap
            aps.append(ap)

            print(
                f"Classe {class_names[c]} | TP: {tp} | FP: {fp} | FN: {fn} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AP@0.5: {ap:.3f}")

        mean_ap = np.mean(aps)
        print(f"\n📊 mAP@0.5 global : {mean_ap:.3f}")
        stats["mAP@0.5"] = mean_ap

        return stats



    def visualize_predictions(self, dataset, predictions, num_images=10, conf_threshold=0.25):
        for i in range(min(num_images, len(predictions))):

            item = dataset[i]
            if len(item) == 4:
                img, labels, image_id, (img_width, img_height) = item
            else:
                img, labels, image_id = item
                img_width, img_height = img.size  # fallback au cas où

            pred = predictions[i]

            # Conversion en numpy si c'est un tensor
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)

            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(img_np)

            def evaluate_debug(self, predictions, dataloader, iou_threshold=0.5, conf_threshold=0.25):
                from ultralytics.utils.metrics import bbox_iou
                import torch

                images, gt_labels_batch, image_ids, sizes = next(iter(dataloader))
                img = images[0]
                labels = gt_labels_batch[0]
                pred = predictions[0]
                img_w, img_h = sizes[0]

                print(f"\n📐 Image size : {img_w}x{img_h}")

                # GT
                gt_boxes = []
                for ann in labels:
                    cls, xc, yc, w, h = ann
                    x1 = (xc - w / 2) * img_w
                    y1 = (yc - h / 2) * img_h
                    x2 = (xc + w / 2) * img_w
                    y2 = (yc + h / 2) * img_h
                    gt_boxes.append([x1, y1, x2, y2])
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

                print(f"✅ Ground Truth boxes :")
                for i, box in enumerate(gt_boxes.tolist()):
                    print(f"  GT {i}: {box}")

                print(f"\n🔴 Predicted boxes :")
                for i, (box, score, cls_pred) in enumerate(zip(pred["boxes"], pred["scores"], pred["classes"])):
                    if score < conf_threshold:
                        continue
                    print(f"  Pred {i} - Score: {score:.2f}, Class: {cls_pred}, Box: {box}")
                    box_tensor = torch.tensor(box, dtype=torch.float32).unsqueeze(0)
                    ious = bbox_iou(box_tensor, gt_boxes)
                    print(f"     → IoUs: {ious.squeeze().tolist()}")

                print("\n✅ Fin du debug 1-frame.")

            # 🟩 Ground Truth boxes
            for label in labels:
                class_id, xc, yc, w, h = label
                x1 = (xc - w / 2) * img_width
                y1 = (yc - h / 2) * img_height
                w_abs = w * img_width
                h_abs = h * img_height

                color = 'green' if class_id == 0 else 'cyan'
                name = 'GT-Person' if class_id == 0 else 'GT-Ball'
                rect = patches.Rectangle((x1, y1), w_abs, h_abs,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, name, color=color,
                        bbox=dict(facecolor='black', alpha=0.5), fontsize=8)

            # 🟥 Predicted boxes
            for box, score, cls in zip(pred["boxes"], pred["scores"], pred["classes"]):
                if score < conf_threshold:
                    continue
                x1, y1, x2, y2 = box
                w_pred = x2 - x1
                h_pred = y2 - y1

                color = 'red' if cls == 0 else 'blue'
                name = 'PRED-Person' if cls == 0 else 'PRED-Ball'
                rect = patches.Rectangle((x1, y1), w_pred, h_pred,
                                         linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1 + 10, f"{name} ({score:.2f})", color=color,
                        bbox=dict(facecolor='white', alpha=0.7), fontsize=8)

            ax.set_title(f"🖼 Image ID : {image_id}")
            ax.axis('off')
            plt.tight_layout()
            plt.show()


