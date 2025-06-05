import json
import os
from collections import Counter
from pprint import pprint

json_path = "../data/SoccerNetGS/train/SNGS-060/Labels-GameState.json"

with open(json_path, "r") as f:
    annotations_json = json.load(f)

# Aperçu global des clés présentes dans chaque annotation
all_keys = set()
missing_bbox = 0
category_counter = Counter()

for ann in annotations_json["annotations"]:
    all_keys.update(ann.keys())

    if "bbox_image" not in ann:
        missing_bbox += 1
        pprint(ann)  # Afficher clairement les annotations sans bbox_image

    category_counter[ann.get("category_id", "no_category_id")] += 1

print("\n✅ Toutes les clés existantes dans les annotations :")
pprint(all_keys)

print(f"\n⚠️ Nombre d'annotations sans 'bbox_image' : {missing_bbox}\n")

print("🔍 Distribution des catégories (category_id) :")
for cat_id, count in category_counter.items():
    print(f"  - category_id {cat_id} : {count} annotations")
