import pandas as pd
import numpy as np

def parse_image_id(id_int):
    id_str = str(id_int).zfill(10)
    set_id = int(id_str[0])
    clip_id = int(id_str[1:4])
    frame_idx = int(id_str[7:10])
    return set_id, clip_id, frame_idx

def load_and_prepare():
    df = pd.read_csv("/Users/jeangrifnee/PycharmProjects/soccernet/evalCalibration/calibr_result/full_TVCalib/projection_errors_per_frame.csv")
    df[["set_id", "clip_id", "frame_idx"]] = df["image_id"].apply(lambda x: pd.Series(parse_image_id(x)))
    return df

def analyze_clip_stability(df):
    # Seuil dynamique pour frames à erreur élevée (ex : top 5% des erreurs du dataset)
    error_threshold = df["avg_error"].quantile(0.95)

    clip_stats = df.groupby("clip_id").agg(
        mean_error=("avg_error", "mean"),
        max_error=("avg_error", "max"),
        std_error=("avg_error", "std"),
        nb_frames=("avg_error", "count"),
        nb_high_error_frames=("avg_error", lambda x: (x > error_threshold).sum())
    ).reset_index()

    # Ratio de frames très mauvaises
    clip_stats["high_error_ratio"] = clip_stats["nb_high_error_frames"] / clip_stats["nb_frames"]

    # Catégorisation des clips
    def categorize(row):
        if row["mean_error"] > 15 and row["std_error"] < 5:
            return "constantly_bad"
        elif row["mean_error"] < 10 and row["high_error_ratio"] > 0.2:
            return "locally_bad"
        elif row["std_error"] > 8:
            return "unstable"
        elif row["mean_error"] < 7 and row["std_error"] < 3:
            return "stable"
        else:
            return "mixed"

    clip_stats["clip_pattern"] = clip_stats.apply(categorize, axis=1)
    return clip_stats

def main():
    df = load_and_prepare()
    clip_stats = analyze_clip_stability(df)

    print("=== Résumé des patterns par clip ===")
    print(clip_stats["clip_pattern"].value_counts())
    print("\nQuelques exemples :")
    print(clip_stats.groupby("clip_pattern").head(3)[["clip_id", "mean_error", "max_error", "std_error", "clip_pattern"]])

if __name__ == "__main__":
    main()


