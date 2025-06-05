import sys
import os
import comet_ml
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolo_model import YOLOModel
from resize import ResizeWithPadding
# Permet d'importer depuis le même dossier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import SoccerNetDataset  # importer ta classe
from torchvision import transforms


def collate_fn(batch):
    images, labels, image_ids, sizes = zip(*batch)
    return list(images), list(labels), list(image_ids), list(sizes)
# Transformation claire : Redimensionne à 640x640 (multiple de 32)
transform = transforms.Compose([
    ResizeWithPadding((640, 640)),
    transforms.ToTensor(),
])
def convert_numpy(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    raise TypeError(f"Type not serializable: {type(o)}")
def visualize_sample(dataset, idx):
    img, labels, image_id = dataset[idx]

    # Convertit le tensor en image PIL si nécessaire
    if hasattr(img, 'numpy'):
        img = img.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    img_height, img_width = img.shape[:2]

    for label in labels:
        class_id, x_center, y_center, w, h = label

        # Coordonnées absolues pour affichage
        bbox_x = (x_center - w / 2) * img_width
        bbox_y = (y_center - h / 2) * img_height
        bbox_w = w * img_width
        bbox_h = h * img_height

        rect = patches.Rectangle(
            (bbox_x, bbox_y), bbox_w, bbox_h,
            linewidth=2, edgecolor='red' if class_id == 0 else 'blue', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bbox_x, bbox_y, 'Player' if class_id == 0 else 'Ball', color='white',
                bbox=dict(facecolor='red' if class_id == 0 else 'blue', alpha=0.5))

    plt.title(f"Image ID : {image_id}")
    plt.axis('off')
    plt.show()
# Définir clairement le chemin relatif vers le dossier data
root_dir = "../data/SoccerNetGS"
model_path = "../pretrained_models/yolo/yolo11l.pt"
model_path_finetuned = "/Users/jeangrifnee/PycharmProjects/soccernet/Finetuning/first_test_finetuning_whith_skip_frame1/fine_tuned/weights/best.pt"
model_name = os.path.splitext(os.path.basename(model_path))[0]

# 📁 Dossier où tu veux enregistrer les résultats
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# 📄 Chemin du fichier à sauvegarder
results_path = os.path.join(results_dir, f"metrics_{model_name}.json")
#train_dataset = SoccerNetDataset(root_dir, "train", transforms=transforms.ToTensor())
#test_dataset  = SoccerNetDataset(root_dir, "test")
comet_ml.login(project_name="finetuning_ball_model2")
#test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
model = YOLOModel(model_path)  # Exemple de modèle pré-entraîné YOLOv8 Nano
""""
predictions, fps = model.inference(dataloader=test_loader, conf_threshold=0.25)
metrics = model.evaluate(predictions, dataloader=test_loader)
# 💾 Sauvegarde en JSON
with open(results_path, "w") as f:
    json.dump(metrics, f, indent=4, default=convert_numpy)

print(f"✅ Résultats sauvegardés dans : {results_path}")
"""
model.train(
    data_yaml_path="data.yaml",
    epochs=50,
    save_dir="finetuning_ball_detection2",
    lr=0.001,
    batch=8,
    img_size=384
)
""""
#model.visualize_predictions(test_dataset, predictions, num_images=10)
# 💾 Sauvegarde en JSON
with open(results_path, "w") as f:
    json.dump(metrics, f, indent=4, default=convert_numpy)

print(f"✅ Résultats sauvegardés dans : {results_path}")

valid_dataset = SoccerNetDataset(root_dir, "challenge", transforms=transforms.ToTensor())
test_dataset  = SoccerNetDataset(root_dir, "test", transforms=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Test rapide pour vérifier que ça marche
for images, labels, image_ids in train_loader:
    print(f"Loaded batch of {len(images)} images")
    break
"""