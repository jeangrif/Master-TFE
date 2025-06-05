import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from model.model import TDEEDModel
from util.dataset import load_classes
from util.eval import evaluate

# === 🔧 Paramètres ===
frame_dir = "/Users/jeangrifnee/PycharmProjects/soccernet/data/SoccerNetGS/train/SNGS-062/img1/"
checkpoint_path = "/Users/jeangrifnee/PycharmProjects/T-DEED/checkpoints/SoccerNetBall/checkpoint_best.pt"
class_file = "data/soccernetball/class.txt"
save_path = "preds_eval_json/"  # dossier où sera écrit le JSON
video_name = "SNGS-062"
clip_len = 100
stride = 1
overlap_len = 75

# === 💻 Device ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Device utilisé : {device}")

# === 📚 Classes ===
class_dict = load_classes(class_file)

# === 📦 Dataset compatible evaluate()
class CustomVideoDataset(Dataset):
    def __init__(self, frame_dir, video_name, clip_len=100, stride=1, overlap_len=75):
        self.frame_dir = frame_dir
        self.video_name = video_name
        self.clip_len = clip_len
        self.stride = stride
        self._stride = stride
        self.overlap_len = overlap_len
        self.hop = (clip_len - overlap_len) * stride
        self._dataset = "soccernetball"

        self.frames = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        self.num_frames = len(self.frames)
        self.num_clips = max(0, (self.num_frames - (clip_len - 1) * stride) // self.hop)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.videos = [(self.video_name, self.num_frames, 25)]  # 25 fps
        self.labels = []  # vide en mode test

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        clip = []
        start_idx = idx * self.hop
        for i in range(self.clip_len):
            frame_idx = start_idx + i * self.stride
            img = Image.open(self.frames[frame_idx]).convert('RGB')
            clip.append(self.transform(img))
        return {
            "video": self.video_name,
            "start": start_idx,
            "frame": torch.stack(clip, dim=0)  # [T, 3, H, W]
        }

# === 🧠 Chargement modèle
class Args: pass
args = Args()
args.dataset = 'soccernetball'
args.modality = 'rgb'
args.clip_len = clip_len
args.crop_dim = None
args.feature_arch = "rny008_gsf"
args.temporal_arch = "ed_sgp_mixer"
args.n_layers = 2
args.sgp_ks = 9
args.sgp_r = 4
args.radi_displacement = 4
args.num_classes = 12
args.pretrain = None
args.model = "SoccerNetBall"

model = TDEEDModel(args=args)
model.device = device
n_classes = [13, 18]
model._model.update_pred_head(n_classes)
model._num_classes = np.array(n_classes).sum()
model._model.to(device)
model._model.eval()
model.load(torch.load(checkpoint_path, map_location=device))

# === 📡 Inference & Post-processing avec evaluate()
dataset = CustomVideoDataset(
    frame_dir=frame_dir,
    video_name=video_name,
    clip_len=clip_len,
    stride=stride,
    overlap_len=overlap_len
)
if __name__ == "__main__":
    evaluate(
        model=model,
        dataset=dataset,
        split="CHALLENGE",
        classes=class_dict,     # ✅ OK maintenant
        save_pred=save_path,    # ✅ remplace save_path → save_pred
        test=True,              # ou False si tu veux les scores
        printed=True
    )

    print(f"✅ Résultats sauvegardés dans {save_path}/{video_name}.json")

