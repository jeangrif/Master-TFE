import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from joblib import Parallel, delayed
import xgboost as xgb
import itertools
import warnings
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings("ignore")

# --- Charger les données ---
df_train = pd.read_csv("features_train.csv")
df_valid = pd.read_csv("features_valid.csv")
q1 = df_valid["reprojection_error"].quantile(0.25)
q2 = df_valid["reprojection_error"].quantile(0.50)
q3 = df_valid["reprojection_error"].quantile(0.75)


# --- Grid de recherche ---
task = "outliers"
if task =="outliers":
    reprojection_thresholds = [10.0, 15.0, 20.0, 25.0]
else:
    reprojection_thresholds = [2,3,4,5]
scale_weights = [1, 5, 10]
decision_thresholds = np.linspace(0.2, 0.9, 3)

# --- Features utilisées ---
exclude = ["image_id", "clip_id", "frame_idx", "split", "reprojection_error", "is_bad"]
features = [col for col in df_train.columns if col not in exclude]

# --- Fonction d’évaluation d’une combinaison ---
def evaluate_combination(reproj_thresh, scale_w, task ):
    df_train_ = df_train.copy()
    df_valid_ = df_valid.copy()
    if task == "outliers":
        df_train_["target"] = (df_train_["reprojection_error"] > reproj_thresh).astype(int)
        df_valid_["target"] = (df_valid_["reprojection_error"] > reproj_thresh).astype(int)
    elif task == "anchor":
        df_train_["target"] = (df_train_["reprojection_error"] < reproj_thresh).astype(int)
        df_valid_["target"] = (df_valid_["reprojection_error"] < reproj_thresh).astype(int)
    else:
        raise ValueError(f"Tâche inconnue : {task}")


    y_train = df_train_["target"]
    y_valid = df_valid_["target"]
    X_train = df_train_[features]
    X_valid = df_valid_[features]
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_w,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_valid)[:, 1]
    df_valid_ = df_valid_.copy()
    df_valid_["predicted_confidence"] = y_proba

    # Corrélation entre confiance et reprojection error (en valeur négative = bon signe)
    corr_spearman, _ = spearmanr(df_valid_["predicted_confidence"], df_valid_["reprojection_error"])
    results = []

    for decision_thresh in decision_thresholds:
        y_pred = (y_proba > decision_thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
        total = tn + fp + fn + tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_good = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        percent_dropped = (fp + tp) / total * 100

        results.append({
            "reprojection_threshold": reproj_thresh,
            "scale_pos_weight": scale_w,
            "decision_threshold": decision_thresh,
            "recall_good": recall,
            "precision_good": precision_good,
            "recall_bad": recall,
            "precision_bad": precision,
            "nb_FP": fp,
            "nb_FN": fn,
            "nb_dropped": fp + tp,
            "spearman_corr": corr_spearman,
            "percent_dropped": percent_dropped,
            "percent_selected": (tp + tn) / total * 100
        })
    if task == "anchor" and reproj_thresh == reprojection_thresholds[0] and scale_w == scale_weights[0]:
        import matplotlib.pyplot as plt

        df_calib = df_valid_.copy()
        df_calib["predicted_confidence"] = y_proba

        # Zoom sur les erreurs de reprojection < 500
        df_zoom = df_calib[df_calib["reprojection_error"] < 500]

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df_zoom["reprojection_error"],
            df_zoom["predicted_confidence"],
            alpha=0.3,
            c=df_zoom["target"],
            cmap="coolwarm",
            label="is_anchor"
        )

        plt.axhline(y=df_zoom["predicted_confidence"].mean(), color='gray', linestyle='--', label='mean proba')
        plt.xlabel("Reprojection Error")
        plt.ylabel("Predicted Confidence (proba)")
        plt.title("Calibration du modèle (Zoom <500)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()



    return results

# --- Parallélisation sur les seuils de reprojection ---
param_grid = list(itertools.product(reprojection_thresholds, scale_weights))
all_results = Parallel(n_jobs=8)(
    delayed(evaluate_combination)(r_thresh, s_weight,task)
    for r_thresh, s_weight in param_grid
)


# --- Aplatir les résultats ---
flat_results = [item for sublist in all_results for item in sublist]
df_results = pd.DataFrame(flat_results)
if task == "outliers":
    df_results.sort_values(by=["recall_bad", "precision_good"], ascending=[False, False], inplace=True)
elif task == "anchor":
    df_results.sort_values(by=["precision_good", "spearman_corr"], ascending=[False, False], inplace=True)

# --- Sauvegarde CSV ---
df_results.to_csv("search_results.csv", index=False)
print("✅ Résultats sauvegardés dans search_results.csv")

# --- Afficher les 10 meilleurs compromis ---
print("\n🔝 Top 10 des configurations :")
print(df_results.head(10).to_string(index=False))



best_config = df_results.iloc[0]
best_thresh = best_config["reprojection_threshold"]
best_weight = best_config["scale_pos_weight"]
df_train_ = df_train.copy()
if task == "anchor":
    df_train_["target"] = (df_train_["reprojection_error"] < best_thresh).astype(int)
else:
    df_train_["target"] = (df_train_["reprojection_error"] > best_thresh).astype(int)

X_train = df_train_[features]
y_train = df_train_["target"]
"""
clf_best = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=best_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    verbosity=0
)

clf_best.fit(X_train, y_train)

# 💾 Sauvegarde du modèle avec le nom de la task
import pickle
model_filename = f"full_features_best_model_{task}.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(clf_best, f)

print(f"\n💾 Modèle sauvegardé dans {model_filename}")
"""