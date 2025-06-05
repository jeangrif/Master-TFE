## Code inspired by https://github.com/ML-KULeuven/socceraction
import os
import warnings
import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from socceraction.spadl import config as spadl_config
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
import socceraction.vaep.formula as vaepformula
import pandas as pd

import matplotsoccer
import matplotlib.pyplot as plt

def analyze_vaep_players_and_actions(A, games, players, spadl_h5, team_filter="Belgium", show_top_actions=True):
    # Ajout d'un compteur d'actions par joueur
    A["count"] = 1

    # Compute each player's number of actions and total VAEP values
    playersR = (
        A[["player_id", "vaep_value", "offensive_value", "defensive_value", "count"]]
        .groupby(["player_id"])
        .sum()
        .reset_index()
    )
    # Add player names
    playersR = playersR.merge(players[["player_id", "nickname", "player_name"]], how="left")
    playersR["player_name"] = playersR[["nickname", "player_name"]].apply(
        lambda x: x.iloc[0] if x.iloc[0] else x.iloc[1], axis=1)
    # Show results
    playersR = playersR[["player_id", "player_name", "vaep_value", "offensive_value", "defensive_value", "count"]]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    cols = ["player_id", "player_name", "vaep_value", "offensive_value", "defensive_value", "count"]
    print(playersR.sort_values("vaep_value", ascending=False)[cols].head(10))
    # 1. Charger les minutes
    with pd.HDFStore(spadl_h5) as store:
        player_games = store["player_games"]

    # 2. Somme des minutes par joueur
    minutes_per_player = player_games.groupby("player_id")["minutes_played"].sum().reset_index()

    # 3. Joindre à ton DataFrame VAEP
    playersR = playersR.merge(minutes_per_player, on="player_id", how="left")

    # 4. Filtrer uniquement ceux qui ont > 900 minutes
    playersR = playersR[playersR["minutes_played"] > 2700]

    # 5. Calculer VAEP / 90 minutes
    playersR["vaep_value_per_90"] = playersR["vaep_value"] / playersR["minutes_played"] * 90

    # 6. Trier et afficher
    playersR = playersR.sort_values("vaep_value_per_90", ascending=False)

    print(playersR[["player_id", "player_name", "vaep_value_per_90", "vaep_value", "minutes_played", "count"]].head(10))

    # === 3. Visualisation des meilleures actions ===
    """"
    if show_top_actions:
        sorted_A = A.sort_values("vaep_value", ascending=False)
        sorted_A = sorted_A[sorted_A.team_name == team_filter]
        sorted_A = sorted_A[~sorted_A.type_name.str.contains("shot")]

        def get_time(period_id, time_seconds):
            m = int((period_id - 1) * 45 + time_seconds // 60)
            s = int(time_seconds % 60)
            return f"{m}m{s}s"

        for j in range(0, 10):
            row = list(sorted_A[j:j+1].itertuples())[0]
            i = row.Index
            a = A[i - 3: i + 2].copy()

            a["player_name"] = a[["nickname", "player_name"]].apply(
                lambda x: x.iloc[0] if pd.notna(x.iloc[0]) and x.iloc[0] != "" else x.iloc[1], axis=1
            )

            g = list(games[games.game_id == a.game_id.values[0]].itertuples())[0]
            game_info = f"{g.game_date} {g.home_team_name} {g.home_score}-{g.away_score} {g.away_team_name}"
            minute = int((row.period_id - 1) * 45 + row.time_seconds // 60)
            print(f"{game_info} {minute}' {row.type_name} {row.player_name}")

            a["scores"] = a.scores.apply(lambda x: "%.3f" % x)
            a["vaep_value"] = a.vaep_value.apply(lambda x: "%.3f" % x)
            a["time"] = a[["period_id", "time_seconds"]].apply(lambda x: get_time(*x), axis=1)

            cols = ["time", "type_name", "player_name", "team_name", "scores", "vaep_value"]
            plt.figure(figsize=(12, 6))  # agrandir la figure
            matplotsoccer.actions(
                a[["start_x", "start_y", "end_x", "end_y"]],
                a.type_name,
                team=a.team_name,
                result=a.result_name == "success",
                label=a[cols],
                labeltitle=cols,
                zoom=False
            )
            plt.show()
    """
    return None

def compute_vaep_from_predictions(spadl_h5, predictions_h5):
    A = []
    with pd.HDFStore(spadl_h5) as spadlstore:
        games = (
            spadlstore["games"]
            .merge(spadlstore["competitions"], how='left')
            .merge(spadlstore["teams"].add_prefix('home_'), how='left')
            .merge(spadlstore["teams"].add_prefix('away_'), how='left')
        )

        players = spadlstore["players"]
        teams = spadlstore["teams"]

    for game in tqdm.tqdm(list(games.itertuples()), desc="Rating actions"):
        actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
        actions = (
            spadl.add_names(actions)
            .merge(players, how="left")
            .merge(teams, how="left")
            .sort_values(["game_id", "period_id", "action_id"])
            .reset_index(drop=True)
        )
        preds = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
        values = vaepformula.value(actions, preds.scores, preds.concedes)
        A.append(pd.concat([actions, preds, values], axis=1))

    A = pd.concat(A).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    return A, games, players

def train_vaep_models_and_predict(X, Y, games, spadl_h5, predictions_h5):
    from xgboost import XGBClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

    Y_hat = pd.DataFrame()
    models = {}

    for col in Y.columns:
        y = Y[col]
        pos_weight = (len(y) - sum(y)) / sum(y)
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            n_jobs=-3,
            verbosity=1,
            enable_categorical=True
        )
        print(f"Training model for: {col}")
        model.fit(X, y)
        models[col] = model

    testX, testY = X, Y

    def evaluate(y, y_hat):
        p = sum(y) / len(y)
        base = [p] * len(y)
        brier = brier_score_loss(y, y_hat)
        print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
        ll = log_loss(y, y_hat)
        print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
        print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

    for col in testY.columns:
        Y_hat[col] = [p[1] for p in models[col].predict_proba(testX)]
        print(f"### Y: {col} ###")
        evaluate(testY[col], Y_hat[col])

    # Générer la colonne "game_id" alignée avec les actions
    A = []
    for game_id in tqdm.tqdm(games.game_id, desc="Loading game ids"):
        Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
        A.append(Ai[["game_id"]])
    game_ids = pd.concat(A).reset_index(drop=True)

    # Grouper par match et sauvegarder, mais ne PAS sauvegarder game_id
    grouped_predictions = pd.concat([game_ids, Y_hat], axis=1).groupby("game_id")
    with pd.HDFStore(predictions_h5) as predictionstore:
        for game_id, df in tqdm.tqdm(grouped_predictions, desc="Saving predictions per game"):
            df = df.reset_index(drop=True)
            # ✅ on enlève game_id pour rester fidèle à la structure attendue
            df = df[Y_hat.columns]
            predictionstore.put(f"game_{int(game_id)}", df, format='table')

    return models


def load_vaep_data(spadl_h5, features_h5, labels_h5, nb_prev_actions=1):
    import socceraction.vaep.features as fs
    import pandas as pd
    import tqdm
    # Charger la table des matchs
    games = pd.read_hdf(spadl_h5, "games")
    """"
    games = games[
        (games["season_id"] == 3) &
        (games["competition_id"] == 43)
        ]
    print("nb of games:", len(games))
    """

    # Définir les features utilisées
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        # fs.bodypart,
        fs.bodypart_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        # fs.time,
        fs.time_delta,
        # fs.actiontype_result_onehot
    ]
    Xcols = fs.feature_column_names(xfns, nb_prev_actions)

    # Charger X et Y
    X, Y = [], []
    for game_id in tqdm.tqdm(games.game_id, desc="Loading features & labels"):
        try:
            Xi = pd.read_hdf(features_h5, f"game_{game_id}")[Xcols]
            Yi = pd.read_hdf(labels_h5, f"game_{game_id}")[["scores", "concedes"]]
            if len(Xi) == len(Yi):
                X.append(Xi)
                Y.append(Yi)
        except Exception as e:
            print(f"Skipping game {game_id}: {e}")
    X = pd.concat(X).reset_index(drop=True)
    Y = pd.concat(Y).reset_index(drop=True)

    return games, X, Y

def getXY(games, Xcols, features_h5, labels_h5):
    X, Y = [], []
    for game_id in tqdm.tqdm(games.game_id, desc="Loading X and Y"):
        try:
            Xi = pd.read_hdf(features_h5, f"game_{game_id}")[Xcols]
            Yi = pd.read_hdf(labels_h5, f"game_{game_id}")[["scores", "concedes"]]
            if len(Xi) == len(Yi):
                X.append(Xi)
                Y.append(Yi)
        except Exception as e:
            print(f"Skipping game {game_id} due to: {e}")
    X = pd.concat(X).reset_index(drop=True)
    Y = pd.concat(Y).reset_index(drop=True)
    return X, Y

def generateLabelFeatures(spadl_h5,features_h5, labels_h5, games):
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        fs.bodypart,
        fs.bodypart_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        fs.time,
        fs.time_delta
    ]

    with pd.HDFStore(spadl_h5) as spadlstore, pd.HDFStore(features_h5) as featurestore:
        for game in tqdm.tqdm(list(games.itertuples()), desc=f"Generating and storing features in {features_h5}"):
            actions = spadlstore[f"actions/game_{game.game_id}"]
            gamestates = fs.gamestates(spadl.add_names(actions), 3)
            gamestates = fs.play_left_to_right(gamestates, game.home_team_id)

            X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
            featurestore.put(f"game_{game.game_id}", X, format='table')

    yfns = [lab.scores, lab.concedes, lab.goal_from_shot]

    with pd.HDFStore(spadl_h5) as spadlstore, pd.HDFStore(labels_h5) as labelstore:
        for game in tqdm.tqdm(list(games.itertuples()), desc=f"Computing and storing labels in {labels_h5}"):
            actions = spadlstore[f"actions/game_{game.game_id}"]
            Y = pd.concat([fn(spadl.add_names(actions)) for fn in yfns], axis=1)
            labelstore.put(f"game_{game.game_id}", Y, format='table')

if __name__ == '__main__':
    datafolder = "data-fifa"
    spadl_h5 = os.path.join(datafolder, "spadl-statsbomb3.h5")
    features_h5 = os.path.join(datafolder, "features3.h5")
    labels_h5 = os.path.join(datafolder, "labels3.h5")
    predictions_h5 = os.path.join(datafolder, "predictions3.h5")

    print("🔁 Chargement des données...")
    games, X, Y = load_vaep_data(spadl_h5, features_h5, labels_h5)


    print("🤖 Entraînement des modèles et génération des prédictions...")
    train_vaep_models_and_predict(X, Y, games, spadl_h5, predictions_h5)

    print("📊 Calcul des valeurs VAEP par action...")
    vaep_df, games, players = compute_vaep_from_predictions(spadl_h5, predictions_h5)

    print("✅ Terminé ! DataFrame final VAEP :", vaep_df.shape)

    stats = analyze_vaep_players_and_actions(
        A=vaep_df,
        games=games,
        players=players,
        spadl_h5=spadl_h5,
        team_filter="Belgium"
    )
