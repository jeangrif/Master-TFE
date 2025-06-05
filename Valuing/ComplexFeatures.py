import pandas as pd
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
class ComplexFeatureGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df["freeze_frame"] = self.df["freeze_frame"].dropna().apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    def compute_numerical_superiority(self):
        """
        Compte le nombre de teammates et opponents situés entre le porteur et le but (x > actor_x).
        """

        def count_players_ahead(freeze):
            #print("Type de freeze :", type(freeze))

            if not isinstance(freeze, list):
                return None, None
            if not freeze:
                return None, None
            #print("🔎 Contenu de freeze_frame :", freeze)
            #quit()
            try:
                actor = next(p for p in freeze if p.get("actor") and p.get("location") not in [None, [None, None]])
                actor_x, actor_y = actor["location"]
            except StopIteration:
                return None, None

            n_teammates = 0
            n_opponents = 0

            for p in freeze:
                loc = p.get("location")
                if not loc or None in loc or p.get("actor"):
                    continue  # on ignore le porteur lui-même
                x, y = loc

                if x > actor_x:
                    if p.get("teammate"):
                        n_teammates += 1
                    else:
                        n_opponents += 1

            return n_teammates, n_opponents

        # Application
        self.df[["n_teammates_ahead", "n_opponents_ahead"]] = self.df["freeze_frame"].apply(
            lambda ff: pd.Series(count_players_ahead(ff))
        )

    def compute_pressure_ratio(self, radius=5.0):
        """
        Calcule la différence de pression autour du porteur avant/après l’action.
        Pression = nombre d'opposants dans un rayon donné autour du porteur.
        """
        pressures = []
        df = self.df.reset_index(drop=True)

        for i in range(len(df)):
            current_row = df.iloc[i]
            current_freeze = current_row["freeze_frame"]

            if not isinstance(current_freeze, list):
                pressures.append(None)
                continue

            # Porté du ballon
            try:
                actor = next(
                    p for p in current_freeze if p.get("actor") and p.get("location") not in [None, [None, None]])
                actor_loc = np.array(actor["location"])
            except StopIteration:
                pressures.append(None)
                continue

            # Défenseurs proches AVANT
            opponents_before = [
                p for p in current_freeze
                if not p.get("teammate") and p.get("location") not in [None, [None, None]]
            ]
            count_before = sum(
                1 for opp in opponents_before
                if np.linalg.norm(np.array(opp["location"]) - actor_loc) <= radius
            )

            # Frame suivante
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                next_freeze = next_row["freeze_frame"]

                if isinstance(next_freeze, list):
                    try:
                        next_actor = next(
                            p for p in next_freeze if p.get("actor") and p.get("location") not in [None, [None, None]])
                        next_loc = np.array(next_actor["location"])
                    except StopIteration:
                        pressures.append(None)
                        continue

                    opponents_after = [
                        p for p in next_freeze
                        if not p.get("teammate") and p.get("location") not in [None, [None, None]]
                    ]
                    count_after = sum(
                        1 for opp in opponents_after
                        if np.linalg.norm(np.array(opp["location"]) - next_loc) <= radius
                    )

                    pressures.append(count_before - count_after)
                else:
                    pressures.append(None)
            else:
                pressures.append(None)

        self.df["pressure_ratio"] = pressures

    def compute_passing_opportunities(self, block_radius=2.0, long_pass_dist=25.0, receiver_safe_radius=5.0):
        """
        Compte les teammates atteignables en passe directe :
        - ligne non bloquée par un opponent
        - ou receveur libre si passe longue
        """

        def point_to_segment_dist(p, a, b):
            """Distance minimale entre point p et segment [a-b]"""
            p, a, b = np.array(p), np.array(a), np.array(b)
            ap = p - a
            ab = b - a
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)

        reachable_counts = []

        for freeze in self.df["freeze_frame"]:
            if not isinstance(freeze, list) or not freeze:
                reachable_counts.append(None)
                continue

            try:
                actor = next(p for p in freeze if p.get("actor") and p.get("location") not in [None, [None, None]])
                actor_loc = np.array(actor["location"])
            except StopIteration:
                reachable_counts.append(None)
                continue

            teammates = [p for p in freeze if
                         p.get("teammate") and not p.get("actor") and p.get("location") not in [None, [None, None]]]
            opponents = [p for p in freeze if not p.get("teammate") and p.get("location") not in [None, [None, None]]]

            count = 0
            for mate in teammates:
                mate_loc = np.array(mate["location"])
                dist = np.linalg.norm(mate_loc - actor_loc)

                # Cas 1 : passe courte ou moyenne, ligne dégagée
                blocked = any(
                    point_to_segment_dist(opp["location"], actor_loc, mate_loc) <= block_radius for opp in opponents)
                if dist <= long_pass_dist and not blocked:
                    count += 1
                    continue

                # Cas 2 : passe longue, receveur libre
                if dist > long_pass_dist:
                    near_defenders = sum(
                        np.linalg.norm(np.array(opp["location"]) - mate_loc) <= receiver_safe_radius for opp in
                        opponents)
                    if near_defenders == 0:
                        count += 1

            reachable_counts.append(count)

        self.df["passing_opportunities"] = reachable_counts

    def compute_freeze_frame_confidences(self, max_opponents_expected=10, max_players_expected=21):
        """
        Calcule deux indices de confiance pour chaque freeze_frame :
        - confidence_opponents : basé sur le nombre de défenseurs visibles
        - confidence_total : basé sur le nombre total de joueurs visibles (hors porteur de balle)
        """

        def count_confidences(freeze):
            if not isinstance(freeze, list):
                return None, None
            opponents = [
                p for p in freeze
                if not p.get("teammate", False) and not p.get("actor", False)
                   and p.get("location") not in [None, [None, None]]
            ]
            all_players = [
                p for p in freeze
                if not p.get("actor", False) and p.get("location") not in [None, [None, None]]
            ]
            return len(opponents), len(all_players)

        counts = self.df["freeze_frame"].apply(count_confidences)
        self.df["confidence_opponents"] = counts.apply(
            lambda tup: round(min(tup[0] / max_opponents_expected, 1.0), 3) if tup[0] is not None else None
        )
        self.df["confidence_total"] = counts.apply(
            lambda tup: round(min(tup[1] / max_players_expected, 1.0), 3) if tup[1] is not None else None
        )

    def get_features(self):
        return self.df

class SequenceSegmenter:
    def __init__(self, rupture_types=None, max_time_gap=8.0, max_distance=40.0):
        self.rupture_types = rupture_types or [2, 3, 4, 5, 6, 12, 13, 22]  # touche, corner, coup franc, etc.
        self.max_time_gap = max_time_gap
        self.max_distance = max_distance

    def is_new_sequence(self, prev, curr):
        # Changement de match ou de période
        if curr["game_id"] != prev["game_id"]:
            return True
        if curr["period_id"] != prev["period_id"]:
            return True

        # Perte de possession (équipe change)
        if curr["team_id"] != prev["team_id"]:
            return True

        # Écart temporel trop grand
        if abs(curr["time_seconds"] - prev["time_seconds"]) > self.max_time_gap:
            return True

        # Action de rupture
        if curr["type_id"] in self.rupture_types:
            return True

        # Distance spatiale trop grande
        dx = curr["start_x"] - prev["end_x"]
        dy = curr["start_y"] - prev["end_y"]
        dist = np.hypot(dx, dy)
        if dist > self.max_distance:
            return True

        return False

    def assign_sequence_ids(self, actions_df):
        actions_df = actions_df.copy()
        sequence_ids = []
        current_id = 0

        for i, row in actions_df.iterrows():
            if i == 0:
                sequence_ids.append(current_id)
                prev_row = row
                continue

            if self.is_new_sequence(prev_row, row):
                current_id += 1
            sequence_ids.append(current_id)
            prev_row = row

        actions_df["sequence_id"] = sequence_ids
        return actions_df


from collections import defaultdict
import numpy as np
import json

class FreezeFrameAggregator:
    def __init__(self, df, max_speed_mps=7.0):
        self.df = df
        self.max_speed_mps = max_speed_mps
        self.estimated_positions = []
        self.df["freeze_frame"] = self.df["freeze_frame"].dropna().apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        # Stats pour le tracking des gardiens
        self.keeper_stats = defaultdict(int)
        self.all_positions = defaultdict(lambda: {"team": defaultdict(list), "opp": defaultdict(list)})

    def process_all_sequences(self):
        grouped = self.df[self.df["freeze_frame"].notna()].groupby("sequence_id")
        for seq_id, group in grouped:
            #print(f"\n🧪 Traitement de la séquence {seq_id} ({len(group)} actions)")
            self.process_sequence(seq_id, group)

        print("\n📊 Statistiques tracking gardien :")
        for k, v in self.keeper_stats.items():
            print(f"  • {k} = {v}")

    def process_sequence(self, seq_id, group):
        group = group.sort_values("time_seconds")
        team_pool = []
        opp_pool = []
        seq_pos = self.all_positions[seq_id]

        def match_players(freeze_players, pool, t, label):
            candidates = []
            matched_freeze = set()
            matched_pool = set()
            keeper_already_seen = any(p.get("is_keeper", False) for p in pool)

            for i, p in enumerate(freeze_players):
                x, y = p["location"]
                for j, player in enumerate(pool):
                    t_last = player["t_last_seen"]
                    dt = t - t_last
                    eps = self.max_speed_mps * dt
                    dist = np.linalg.norm(np.array([x, y]) - np.array(player["location"]))
                    candidates.append((i, j, dist, eps, dt))

            candidates.sort(key=lambda tup: tup[2])

            for i, j, dist, eps, dt in candidates:
                if i in matched_freeze or j in matched_pool:
                    continue

                freeze_is_keeper = freeze_players[i].get("keeper", False)
                pool_is_keeper = pool[j].get("is_keeper", False)

                #if freeze_is_keeper or pool_is_keeper:
                    #print(f"\n🧤 Tentative de matching gardien (freeze idx {i} → pool[{j}])")
                    #print(f"   • Dist = {dist:.2f}, Eps = {eps:.2f}, dt = {dt:.2f}")
                    #print(f"   • pool[{j}] keeper = {pool_is_keeper}, freeze[{i}] keeper = {freeze_is_keeper}")
                    #print(f"   • Match possible ? {'✅ OUI' if dist <= eps else '❌ NON'}")

                if dist <= eps:
                    # Évaluation erreurs spécifiques gardien
                    if freeze_is_keeper and pool_is_keeper:
                        self.keeper_stats["match_correct_keeper"] += 1
                    elif freeze_is_keeper and not pool_is_keeper:
                        self.keeper_stats["bad_match_keeper_to_player"] += 1
                    elif not freeze_is_keeper and pool_is_keeper:
                        self.keeper_stats["bad_match_player_to_keeper"] += 1

                    pool[j]["location"] = freeze_players[i]["location"]
                    pool[j]["t_last_seen"] = t
                    pool[j]["is_keeper"] = freeze_is_keeper
                    pool[j]["just_updated"] = True  # ← temporaire pour marquer ceux mis à jour

                    matched_freeze.add(i)
                    matched_pool.add(j)

                    #if freeze_is_keeper or pool_is_keeper:
                        #print(f"✅ Match finalisé : freeze[{i}] → pool[{j}]")
                    continue

            for i, p in enumerate(freeze_players):
                if i in matched_freeze:
                    continue

                is_keeper = p.get("keeper", False)
                if is_keeper:
                    if not keeper_already_seen:
                        self.keeper_stats["match_correct_keeper"] += 1  # premier ajout = succès
                    else:
                        self.keeper_stats["keeper_added_to_pool"] += 1  # gardien ajouté à nouveau = erreur
                    #print(f"\n🧤 ➕ Aucun match gardien trouvé → ajout dans le pool")

                #print(f"➕ [{label}] Nouveau joueur ajouté — aucune correspondance trouvée")
                pool.append({
                    "location": p["location"],
                    "t_last_seen": t,
                    "is_keeper": is_keeper
                })

        for i, (_, row) in enumerate(group.iterrows()):
            t_curr = row["time_seconds"]
            freeze = row["freeze_frame"]
            if not isinstance(freeze, list):
                continue

            teammates = [p for p in freeze if
                         p.get("teammate") and "location" in p and p["location"] not in [None, [None, None]]]
            opponents = [p for p in freeze if
                         p.get("teammate") is False and not p.get("actor") and "location" in p and p[
                             "location"] not in [None, [None, None]]]

            #print(f"\n⏱️ Action {i} à {t_curr:.2f}s")
            #print(f"👥 Freeze frame → {len(teammates)} teammates, {len(opponents)} opponents observés")
            #print(f"🔎 Total joueurs valides : {len(teammates) + len(opponents)}")

            match_players(teammates, team_pool, t_curr, "team")
            match_players(opponents, opp_pool, t_curr, "opp")
            for label, pool in [("team", team_pool), ("opp", opp_pool)]:
                for j, player in enumerate(pool):
                    if player.pop("just_updated", False):
                        pos_entry = {
                            "t": t_curr,
                            "location": player["location"],
                            "visible": True,
                            "is_keeper": player.get("is_keeper", False)
                        }
                        seq_pos[label][j].append(pos_entry)

            #print(f"📦 Pool team: {len(team_pool)} joueurs — Pool opp: {len(opp_pool)} joueurs")

        #print("\n✅ Résumé final :")
        #print(f"Team: {len(team_pool)} joueurs estimés")
        for i, p in enumerate(team_pool):
            keeper = "🧤" if p.get("is_keeper") else ""
            #print(f"  [{i}] → pos = {p['location']} | t_last_seen = {p['t_last_seen']:.2f}s {keeper}")

        #print(f"Opp: {len(opp_pool)} joueurs estimés")
        for i, p in enumerate(opp_pool):
            keeper = "🧤" if p.get("is_keeper") else ""
            #print(f"  [{i}] → pos = {p['location']} | t_last_seen = {p['t_last_seen']:.2f}s {keeper}")

class ComputeComplexFeature:
    def __init__(self, df, radius=5.0):
        self.df = df.copy()
        self.df["freeze_frame"] = self.df["freeze_frame"].dropna().apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        self.radius = radius

    def compute_pressure_reduction(self):
        results = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            freeze = row["freeze_frame"]
            if not isinstance(freeze, list):
                results.append(None)
                continue

            # Trouver le porteur
            try:
                actor = next(p for p in freeze if p.get("actor") and p.get("location") not in [None, [None, None]])
                actor_loc = np.array(actor["location"])
            except StopIteration:
                results.append(None)
                continue

            # Défenseurs visibles avec une position
            defenders = [
                p for p in freeze
                if not p.get("teammate") and p.get("location") not in [None, [None, None]]
            ]
            if not defenders:
                results.append(0.0)
                continue

            dists = [np.linalg.norm(np.array(d["location"]) - actor_loc) for d in defenders]
            close_def = sum(1 for d in dists if d <= self.radius)
            density = close_def / len(defenders)
            results.append(density)

        # Transformer en différence entre t and t+1
        pressure_diff = [None]
        for i in range(1, len(results)):
            if results[i - 1] is None or results[i] is None:
                pressure_diff.append(None)
            else:
                pressure_diff.append(results[i - 1] - results[i])  # Start - End

        return pressure_diff
