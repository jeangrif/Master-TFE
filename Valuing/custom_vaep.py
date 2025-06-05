# custom_vaep.py
from ComplexFeatures import SequenceSegmenter,ComplexFeatureGenerator,FreezeFrameAggregator, ComputeComplexFeature
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt
def plot_pressure_evolution(df):
    """
    Trace l’évolution du pressure_ratio au fil du temps pour toutes les actions.
    """
    import matplotlib.pyplot as plt

    df_plot = df.copy()
    df_plot = df_plot[df_plot["pressure_ratio"].notna()]

    if df_plot.empty:
        print("❌ Aucune donnée à afficher (pressure_ratio manquant).")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(df_plot["time_seconds"], df_plot["pressure_ratio"], marker='o', linestyle='-')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # ligne 0 pour visuellement séparer pression montée/descente
    plt.title("Évolution du Pressure Ratio")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Pressure Ratio (avant - après)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_frame_0(df):
    """
    Affiche les frames 0 à 4 avec teammates, opponents, actor,
    le type_id en haut à gauche, et le temps de l'action en haut au centre.
    """


    for i in range(5):
        row = df.iloc[i]

        freeze_frame = row["freeze_frame"]
        if isinstance(freeze_frame, str):
            try:
                freeze_frame = json.loads(freeze_frame)
            except Exception:
                print(f"⚠️ Erreur de parsing JSON à l'index {i}.")
                continue

        x_vals = []
        y_vals = []
        colors = []
        labels = []

        for player in freeze_frame:
            loc = player.get("location")
            if loc and None not in loc:
                x, y = loc
                x_vals.append(x)
                y_vals.append(y)

                if player.get("actor", False):
                    colors.append('red')  # 🔴 Actor
                    labels.append('Actor')
                elif player.get("teammate", False):
                    colors.append('blue')  # 🔵 Teammate
                    labels.append('Teammate')
                else:
                    colors.append('green')  # 🟢 Opponent
                    labels.append('Opponent')

        pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
        fig, ax = pitch.draw(figsize=(10, 7))

        # Scatter des joueurs
        scatter = ax.scatter(x_vals, y_vals, c=colors, edgecolors='black', s=100)

        # Labels des joueurs
        for j, label in enumerate(labels):
            ax.text(x_vals[j] + 1, y_vals[j] + 1, label, fontsize=8)

        # Infos textuelles : type_id (haut gauche) + time_seconds (haut centre)
        type_id = row.get("type_id", None)
        time_sec = row.get("time_seconds", None)
        ax.text(5, 75, f"type_id: {type_id}", fontsize=12, color='black', ha='left', va='top')
        ax.text(60, 75, f"time: {time_sec:.1f}s", fontsize=12, color='gray', ha='center', va='top')

        plt.title(f"Freeze Frame {i}", fontsize=14)
        plt.show()




def print_freeze_frame_bounds(df):
    x_vals, y_vals = [], []

    for ff in df["freeze_frame"]:
        if isinstance(ff, str):
            try:
                ff = json.loads(ff)
            except Exception:
                continue
        if not isinstance(ff, list):
            continue

        for p in ff:
            loc = p.get("location")
            if loc and None not in loc:
                x_vals.append(loc[0])
                y_vals.append(loc[1])

    if not x_vals or not y_vals:
        print("❌ Aucune coordonnée valide trouvée dans les freeze_frame.")
    else:
        print(f"📏 X : min = {min(x_vals):.2f}, max = {max(x_vals):.2f}")
        print(f"📏 Y : min = {min(y_vals):.2f}, max = {max(y_vals):.2f}")


def analyze_freeze_frames(df):
    """
    Analyse rapide des freeze_frames dans un DataFrame SPADL enrichi.
    Gère les freeze_frame au format string (après chargement Parquet).
    """
    def parse_and_validate(ff):
        if isinstance(ff, list):
            return ff
        try:
            parsed = json.loads(ff)
            if isinstance(parsed, list) and all(isinstance(p, dict) for p in parsed):
                return parsed
        except Exception:
            pass
        return None

    def count_players(ff):
        return len(ff) if ff else 0

    def has_actor(ff):
        return any(p.get("actor", False) for p in ff) if ff else False

    df = df.copy()
    df["parsed_ff"] = df["freeze_frame"].apply(parse_and_validate)
    df["ff_valid"] = df["parsed_ff"].apply(lambda x: x is not None)
    df["nb_players_visible"] = df["parsed_ff"].apply(count_players)
    df["has_actor"] = df["parsed_ff"].apply(has_actor)

    total = len(df)
    valid = df["ff_valid"].sum()
    actor_visible = df["has_actor"].sum()
    avg_players = df.loc[df["ff_valid"], "nb_players_visible"].mean()

    print(f"✅ {valid} / {total} actions ont un freeze_frame valide")
    print(f"👤 {actor_visible} / {total} actions ont le porteur visible")
    print(f"👥 Nombre moyen de joueurs visibles (sur freeze_frame valides) : {avg_players:.2f}")

def get_eligible_game_ids(freeze_folder, threshold=0.1):
    eligible_ids = []
    for file in os.listdir(freeze_folder):
        if file.endswith(".json"):
            game_id = int(file.replace(".json", ""))
            try:
                with open(os.path.join(freeze_folder, file), "r") as f:
                    data = json.load(f)
                total = len(data)
                with_ff = sum(1 for event in data if event.get("freeze_frame"))
                if total > 0 and (with_ff / total) >= threshold:
                    eligible_ids.append(game_id)
            except json.JSONDecodeError:
                print(f"⚠️  JSONDecodeError dans le fichier : {file}, ignoré.")
    return eligible_ids

def load(freeze_folder: str, spadl_file: str) -> pd.DataFrame:
    """
    Charge les actions SPADL et leurs freeze_frames associées
    pour les matchs avec une couverture > 80%
    """
    # 1. Charger les actions SPADL
    with pd.HDFStore(spadl_file, mode="r") as store:
        keys = [k for k in store.keys() if k.startswith("/actions/game_")]
        dfs = [store[k] for k in keys]
        actions_df = pd.concat(dfs, ignore_index=True)
        print("Colonnes disponibles :", actions_df.columns.tolist())
        # 2. Ajouter home_team_id à chaque action via merge
        if "/games" in store:
            games_df = store["/games"]
            actions_df = actions_df.merge(games_df[["game_id", "home_team_id"]], on="game_id", how="left")
            print("✅ Colonne 'home_team_id' ajoutée.")
        else:
            print("⚠️ Le fichier HDF ne contient pas /games. Impossible d’ajouter home_team_id.")


    # 2. Identifier les game_id valides
    eligible_ids = get_eligible_game_ids(freeze_folder)

    # 3. Filtrer les actions SPADL
    actions_df = actions_df[actions_df["game_id"].isin(eligible_ids)].copy()

    # 4. Construire un mapping event_uuid → freeze_frame
    freeze_map = {}
    for file in tqdm(os.listdir(freeze_folder), desc="Loading freeze frames"):
        if not file.endswith(".json"):
            continue
        game_id = int(file.replace(".json", ""))
        if game_id not in eligible_ids:
            continue
        with open(os.path.join(freeze_folder, file), "r") as f:
            events = json.load(f)
        for event in events:
            uuid = event.get("event_uuid")
            ff = event.get("freeze_frame", None)
            if uuid and ff:
                freeze_map[uuid] = ff

    # 5. Ajouter une colonne 'freeze_frame' à actions_df
    actions_df["freeze_frame"] = actions_df["original_event_id"].map(freeze_map)

    return actions_df

if __name__ == '__main__':
    #df = load(freeze_folder="data/three-sixty", spadl_file="data-fifa/spadl-statsbomb.h5")
    #df["freeze_frame"] = df["freeze_frame"].apply(json.dumps)
    #df.to_parquet("data-fifa/features_complex2.parquet", index=False)
    #quit()
    df = pd.read_parquet("data-fifa/features_complex2.parquet")
    cf = ComplexFeatureGenerator(df)
    cf.compute_numerical_superiority()
    cf.compute_passing_opportunities()
    cf.compute_pressure_ratio()
    cf.compute_freeze_frame_confidences()
    df_features = cf.get_features()
    df_features.to_parquet("data-fifa/features_complex2_fulll.parquet", index=False)
    #feature_gen = ComplexFeatureGenerator(df)
    #feature_gen.compute_numerical_superiority()
    #df_result = feature_gen.get_features()
    #print(df_result[["n_teammates_ahead", "n_opponents_ahead"]].head())
    #print(df.columns)
    #plot_frame_0(df)
    quit()
    import matplotlib.pyplot as plt

    # Supposons que tu as une colonne 'pressure_reduction' dans ton DataFrame
    values = df["pressure_reduction"].dropna()

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=30, edgecolor='black')
    plt.title("Distribution de la Pressure Reduction")
    plt.xlabel("Δ Pression autour du porteur (défenseurs dans le rayon)")
    plt.ylabel("Nombre d'actions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()







    """
    # 2. Segmenter en séquences
    segmenter = SequenceSegmenter()
    df["sequence_id"] = None  # Init colonne
    sequence_id = 0

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        if segmenter.is_new_sequence(prev, curr):
            sequence_id += 1
        df.at[i, "sequence_id"] = sequence_id

    df["sequence_id"] = df["sequence_id"].astype("Int64")
    sampled_ids = df["sequence_id"].dropna().unique()
    selected_ids = np.random.choice(sampled_ids, size=8, replace=False)

    # Garder uniquement les lignes correspondant à ces séquences
    subset = df[df["sequence_id"].isin(selected_ids)].copy()

    # Créer l'agrégateur avec ce sous-ensemble
    aggregator = FreezeFrameAggregator(subset)
    aggregator.process_all_sequences()
    for seq_id in list(aggregator.all_positions.keys())[:8]:
        print(f"\n📌 Timeline des joueurs pour la séquence {seq_id} :")
        for team_label in ["team", "opp"]:
            print(f"\n🔹 Équipe : {team_label}")
            for pid, entries in aggregator.all_positions[seq_id][team_label].items():
                print(f"  Joueur {pid} → {len(entries)} frames")
                for e in entries:
                    print(f"    - t={e['t']:.2f}s, pos={e['location']}, visible={e['visible']}, keeper={e['is_keeper']}")


    quit()

    # Fin de la séquence 42
    seq_0 = df[df["sequence_id"] == 0]
    print("🔚 Fin de la séquence 0 :")
    print(seq_0[["game_id", "period_id", "time_seconds", "team_id", "type_id"]].head(13))

    # Début de la séquence 43
    seq_1 = df[df["sequence_id"] == 1]
    print("\n🔜 Début de la séquence 1 :")
    print(seq_1[["game_id", "period_id", "time_seconds", "team_id", "type_id"]].head(13))

    # Début de la séquence 43
    seq_2 = df[df["sequence_id"] == 2]
    print("\n🔜 Début de la séquence 2 :")
    print(seq_2[["game_id", "period_id", "time_seconds", "team_id", "type_id"]].head(10))

    # Début de la séquence 43
    seq_3 = df[df["sequence_id"] == 3]
    print("\n🔜 Début de la séquence 3 :")
    print(seq_3[["game_id", "period_id", "time_seconds", "team_id", "type_id"]].head(10))

    # Début de la séquence 43
    seq_4 = df[df["sequence_id"] == 4]
    print("\n🔜 Début de la séquence 4 :")
    print(seq_4[["game_id", "period_id", "time_seconds", "team_id", "type_id"]].head(10))
    print(df[df["sequence_id"] == 1]["freeze_frame"].notna().sum())
    quit()
    aggregator = FreezeFrameAggregator(df)
    aggregator.process_all_sequences()
    summary_df = aggregator.get_sequence_summary()
    print(summary_df.head())
    """

