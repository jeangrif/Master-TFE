import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import pandas as pd


def analyze_projection_errors(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["error"])
    errors = df["error"]

    # Quartiles et percentiles
    q1 = errors.quantile(0.25)
    q2 = errors.quantile(0.50)  # Médiane
    q3 = errors.quantile(0.75)
    q4 = errors.max()
    iqr = q3 - q1

    print("📊 STATISTIQUES SUR LES ERREURS DE PROJECTION")
    print("------------------------------------------------")
    print(f"🔢 Nombre total de points             : {len(errors)}")
    print(f"📏 Moyenne                            : {errors.mean():.4f} m")
    print(f"📈 Médiane (Q2)                       : {q2:.4f} m")
    print(f"📉 Écart-type                         : {errors.std():.4f} m")
    print(f"📦 Minimum                            : {errors.min():.4f} m")
    print(f"📦 Maximum (Q4)                       : {q4:.4f} m")
    print("------------------------------------------------")
    print(f"🔹 1er quartile (Q1, 25%)             : {q1:.4f} m")
    print(f"🔸 3e quartile (Q3, 75%)              : {q3:.4f} m")
    print(f"📐 IQR (Q3 - Q1)                      : {iqr:.4f} m")
    print("------------------------------------------------")
    print(f"⚠️  Nb erreurs > 50m                  : {(errors > 50).sum()}")
    print(f"⚠️  Nb erreurs > 100m                 : {(errors > 100).sum()}")
    print(f"🚨 Nb erreurs > 200m                 : {(errors > 200).sum()}")
    print("------------------------------------------------")

    print("🚨 Top 5 plus grosses erreurs :")
    print(df.sort_values("error", ascending=False).head(5).to_string(index=False))




file_path = "/Users/jeangrifnee/PycharmProjects/soccernet/evalCalibration/projection_per_frame.csv"
analyze_projection_errors(file_path)
