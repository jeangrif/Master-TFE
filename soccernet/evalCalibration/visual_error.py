import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CameraFeatureExtractor import CameraFeatureExtractor

def plot_comparaison_clips(df, clips, seuil=25):
    plt.figure(figsize=(14, 5))

    for clip_id in clips:
        clip_df = df[df["clip_id"] == clip_id].sort_values("frame_idx")
        plt.plot(
            clip_df["frame_idx"],
            clip_df["reprojection_error"],
            label=f"Clip {clip_id}",
            marker='o', markersize=3, linewidth=1
        )

    plt.axhline(seuil, color='red', linestyle='--', label=f"Threshold = {seuil}")
    plt.title("Comparison of reprojection error across different clips")
    plt.xlabel("Frame index")
    plt.ylabel("Reprojection error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# === Fonction principale d'analyse des erreurs de calibration === #
def analyse_erreurs_camera(df, seuil=20):
    # Copie de sécurité
    df = df.copy()

    # Vérification de la colonne
    if "reprojection_error" not in df.columns:
        raise ValueError("La colonne 'reprojection_error' est absente du DataFrame.")

    # Conversion et statistiques
    df["reprojection_error"] = pd.to_numeric(df["reprojection_error"], errors="coerce")
    total_frames = len(df)
    nb_nan = df["reprojection_error"].isna().sum()
    print ("nb_nannnnn",nb_nan)
    df = df.dropna(subset=["reprojection_error"])

    print("========== STATISTIQUES GLOBALES ==========")
    print(f"Nombre total de frames       : {total_frames}")
    print(f"Nombre de NaN                : {nb_nan} ({100 * nb_nan / total_frames:.2f}%)")
    print(f"Moyenne                      : {df['reprojection_error'].mean():.2f}")
    print(f"Médiane                      : {df['reprojection_error'].median():.2f}")
    print(f"Min                          : {df['reprojection_error'].min():.2f}")
    print(f"Max                          : {df['reprojection_error'].max():.2f}")
    print(f"Écart-type (std)             : {df['reprojection_error'].std():.2f}")
    print(f"Q1 (25e percentile)          : {df['reprojection_error'].quantile(0.25):.2f}")
    print(f"Q3 (75e percentile)          : {df['reprojection_error'].quantile(0.75):.2f}")
    print(f"P5                           : {df['reprojection_error'].quantile(0.05):.2f}")
    print(f"P95                          : {df['reprojection_error'].quantile(0.95):.2f}")

    """

    # Histogramme + Boxplot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df['reprojection_error'], bins=40, kde=True)
    plt.axvline(seuil, color='r', linestyle='--', label=f"Seuil {seuil}px")
    plt.title("Distribution des erreurs de reprojection")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['reprojection_error'])
    plt.title("Boxplot des erreurs")
    plt.tight_layout()
    plt.show()

    # Ajout d'une colonne booléenne : erreur > seuil
    df["high_error"] = df["reprojection_error"] > seuil

    # Analyse par clip
    clip_summary = df.groupby("clip_id").agg(
        total_frames=("image_id", "count"),
        high_error_frames=("high_error", "sum"),
        avg_error=("reprojection_error", "mean")
    )
    clip_summary["percent_high"] = 100 * clip_summary["high_error_frames"] / clip_summary["total_frames"]
    clip_summary = clip_summary.sort_values("percent_high", ascending=False)

    print("\n========== CLIPS AVEC ERREURS ÉLEVÉES ==========")
    print(clip_summary[clip_summary["percent_high"] > 10].head())

    # Barplot des clips les plus touchés
    plt.figure(figsize=(12, 4))
    sns.barplot(data=clip_summary.reset_index(), x="clip_id", y="percent_high")
    plt.xticks(rotation=90)
    plt.title(f"% de frames avec erreur > {seuil}px par clip")
    plt.ylabel("Pourcentage")
    plt.tight_layout()
    plt.show()
    """
    return df #, clip_summary


# === Analyse détaillée sur un clip particulier === #
def analyse_clip_temporel(df, clip_id, seuil=20):
    clip_df = df[df["clip_id"] == clip_id].sort_values("frame_idx")
    errors = clip_df["reprojection_error"].values
    frames = clip_df["frame_idx"].values

    # Plot des erreurs frame par frame
    plt.figure(figsize=(12, 4))
    plt.plot(frames, errors, marker='o')
    plt.axhline(seuil, color="red", linestyle="--", label=f"Seuil = {seuil}px")
    plt.title(f"Erreur de reprojection — Clip {clip_id}")
    plt.xlabel("Frame index")
    plt.ylabel("Erreur")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Séquences consécutives > seuil
    high = errors > seuil
    sequences = []
    count = 0
    for val in high:
        if val:
            count += 1
        elif count > 0:
            sequences.append(count)
            count = 0
    if count > 0:
        sequences.append(count)

    print(f"Nombre de séquences > {seuil}px : {len(sequences)}")
    if sequences:
        print(f"Longueur moyenne : {np.mean(sequences):.2f}")
        print(f"Max : {np.max(sequences)}")

# === Exemple d’utilisation === #
path ="/Users/jeangrifnee/PycharmProjects/soccernet/outputs/sn-gamestate/2025-03-25/Valid_Calib/calib_pred.json"

# 1. Extraire les features
extractor = CameraFeatureExtractor(path, "gt_valid_calibration.json", split="valid")
extractor.debug_info()


df_features = extractor.build()
gt_raw = extractor.gt
preds_raw = extractor.preds
    # Détection des frames NaN
# 2. Détection des frames NaN
nan_ids = []
for image_id, data in preds_raw.items():
    if image_id not in gt_raw:
        continue
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

print("Nb de frame avec Camera params NaN", len(nan_ids))

# 3. Calcul du P75
p95 = df_features["reprojection_error"].quantile(0.95)
print(f"P95 utilisé comme pénalité : {p95:.2f}")

# 4. Construction du DataFrame pénalisé
df_nan = pd.DataFrame({
    "image_id": nan_ids
})
df_nan["clip_id"] = df_nan["image_id"].apply(CameraFeatureExtractor.extract_clip_id)
df_nan["frame_idx"] = df_nan["image_id"].apply(CameraFeatureExtractor.extract_frame_idx)
df_nan["split"] = "valid"
df_nan["reprojection_error"] = p95
df_nan["penalized"] = True

# Ajouter penalized=False pour les frames valides
df_features["penalized"] = False

# 5. Fusion des deux
df_all = pd.concat([df_features, df_nan], ignore_index=True)

# 6. Analyse complète
df_clean = analyse_erreurs_camera(df_all, seuil=20)


seuil = 25
# Frame considérée comme critique si NaN pénalisé ou erreur > seuil
df_all["critique"] = df_all["penalized"] | (df_all["reprojection_error"] > seuil)

# ---- Q1 : Répartition des frames critiques par clip ---- #
clip_crit = df_all.groupby("clip_id").agg(
    total_frames=("image_id", "count"),
    nb_critiques=("critique", "sum")
)
clip_crit["pct_critiques"] = 100 * clip_crit["nb_critiques"] / clip_crit["total_frames"]

# === Distribution of critical frames per clip === #
plt.figure(figsize=(10, 4))
sns.histplot(clip_crit["pct_critiques"], bins=30, kde=True)
plt.xlabel("% of critical frames per clip")
plt.title("Distribution of critical frames across clips")
plt.grid(True)

# === Prepare stats text === #
stats_text = (
    f"Total clips         : {len(clip_crit)}\n"
    f"Mean                : {clip_crit['pct_critiques'].mean():.2f}%\n"
    f"Median              : {clip_crit['pct_critiques'].median():.2f}%\n"
    f"Min                 : {clip_crit['pct_critiques'].min():.2f}%\n"
    f"Max                 : {clip_crit['pct_critiques'].max():.2f}%\n"
    f"P90                 : {clip_crit['pct_critiques'].quantile(0.90):.2f}%\n"
    f"Clips with 0 crit.  : {(clip_crit['nb_critiques'] == 0).sum()}"
)

# === Display stats in the top-right === #
plt.gca().text(
    0.98, 0.98, stats_text,
    fontsize=9,
    ha='right', va='top',
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
)

plt.tight_layout()

# ---- Q2 + Q3 : Analyse temporelle par clip ---- #
def analyse_clip_patterns(df, clip_id, seuil=20):
    clip_df = df[df["clip_id"] == clip_id].sort_values("frame_idx")
    errors = clip_df["reprojection_error"].values
    penalized = clip_df["penalized"].values
    frames = clip_df["frame_idx"].values

    # Frames critiques : penalized ou > seuil
    is_critique = penalized | (errors > seuil)

    # Identifier les séquences consécutives critiques
    sequences = []
    buffer = []
    for idx, val in enumerate(is_critique):
        if val:
            buffer.append(frames[idx])
        elif buffer:
            sequences.append(buffer)
            buffer = []
    if buffer:
        sequences.append(buffer)

    print(f"\nClip {clip_id} — Nombre de séquences critiques : {len(sequences)}")
    for i, seq in enumerate(sequences[:5]):
        print(f"  Séquence {i+1} : {len(seq)} frames — frames {seq[0]} à {seq[-1]}")

    # Analyse : y a-t-il des "resets" de l’erreur entre les séquences ?
    for i in range(len(sequences) - 1):
        end_current = sequences[i][-1]
        start_next = sequences[i + 1][0]
        intermediate = clip_df[
            (clip_df["frame_idx"] > end_current) &
            (clip_df["frame_idx"] < start_next)
        ]
        if not intermediate.empty:
            min_error = intermediate["reprojection_error"].min()
            print(f"  ➤ Entre séquence {i+1} et {i+2}, erreur min = {min_error:.2f}")

# Exemple : analyser manuellement les clips les plus touchés
worst_clips = clip_crit.sort_values("pct_critiques", ascending=False).head(3).index.tolist()
#for cid in worst_clips:
    #analyse_clip_patterns(df_all, clip_id=cid, seuil=20)
# 3. Analyser un clip en profondeur
# analyse_clip_temporel(df_clean, clip_id=summary.index[0], seuil=20)
# 1. Clip critique extrême (le pire)
clip_extreme = clip_crit.sort_values("pct_critiques", ascending=False).iloc[0].name

# 2. Clip mauvais mais pas extrême (entre 20% et 30% de frames critiques)
clip_bad = clip_crit[(clip_crit["pct_critiques"] > 20) & (clip_crit["pct_critiques"] < 30)].iloc[0].name

# 3. Clip médian (proche de la médiane de distribution)
median_pct = clip_crit["pct_critiques"].median()
clip_median = clip_crit.iloc[(clip_crit["pct_critiques"] - median_pct).abs().argsort()].iloc[0].name

# 4. Clip bon (moins de 2% de frames critiques)
clip_good = clip_crit[clip_crit["pct_critiques"] < 2].iloc[0].name

# Afficher la sélection
print("\nSelected clips for comparison:")
print(f"- Extreme case clip : {clip_extreme}")
print(f"- Bad (non-extreme) : {clip_bad}")
print(f"- Median case clip  : {clip_median}")
print(f"- Good case clip    : {clip_good}")

# Liste à passer dans le plot
clips_to_plot = [clip_extreme, clip_bad, clip_median, clip_good]
#plot_comparaison_clips(df_all, clips_to_plot, seuil=25)

from collections import Counter

# Créer un tableau avec toutes les longueurs de séquences critiques
sequence_lengths = []

for clip_id in df_all["clip_id"].unique():
    clip_df = df_all[df_all["clip_id"] == clip_id].sort_values("frame_idx")
    is_critique = clip_df["penalized"] | (clip_df["reprojection_error"] > seuil)

    count = 0
    for val in is_critique:
        if val:
            count += 1
        elif count > 0:
            sequence_lengths.append(count)
            count = 0
    if count > 0:
        sequence_lengths.append(count)

# Analyse statistique
print("\n===== Global stats on critical sequences =====")
print(f"Total number of critical sequences : {len(sequence_lengths)}")
print(f"Average sequence length            : {np.mean(sequence_lengths):.2f} frames")
print(f"Median sequence length             : {np.median(sequence_lengths):.2f} frames")
print(f"Max sequence length                : {np.max(sequence_lengths)} frames")
print(f"Sequences > 10 frames              : {(np.array(sequence_lengths) > 10).sum()}")
print(f"Sequences > 25 frames              : {(np.array(sequence_lengths) > 25).sum()}")

# Distribution of sequence lengths (optional)
plt.figure(figsize=(10, 4))
sns.histplot(sequence_lengths, bins=30, kde=True)
plt.title("Distribution of critical sequence lengths")
plt.xlabel("Sequence length (in frames)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()