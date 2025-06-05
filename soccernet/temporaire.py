import os
from glob import glob

LABEL_ROOT = "dataset_ball_only/labels"  # dossier parent contenant train/ et valid/

for split in ["train", "valid"]:
    label_dir = os.path.join(LABEL_ROOT, split)
    txt_files = glob(os.path.join(label_dir, "*.txt"))

    print(f"🔧 Correction des labels dans {label_dir} ({len(txt_files)} fichiers)...")

    for txt_path in txt_files:
        new_lines = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == "1.000000":  # balle devient 0
                    parts[0] = "0.000000"
                new_lines.append(" ".join(parts))

        with open(txt_path, "w") as f:
            f.write("\n".join(new_lines))

print("✅ Correction terminée pour train et valid.")