"""
Análisis exploratorio del conjunto de entrenamiento.

Genera figuras y tablas descriptivas en figures/ y results/. Pensado para
ejecutarse una sola vez al principio del proyecto y consultar los gráficos
desde la memoria escrita posteriormente.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import TRAIN_CSV, FIG_DIR, RES_DIR, TARGET, ID_COL, DROP_COLS

sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    summary = {}
    summary["shape"] = list(df.shape)
    summary["missing"] = df.isna().sum().to_dict()
    summary["target_distribution"] = df[TARGET].value_counts(normalize=True).to_dict()

    # Resumen numérico clásico.
    df.describe(include="all").to_csv(RES_DIR / "eda_describe.csv")

    # Distribución del target.
    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET].value_counts().plot(kind="bar", ax=ax, color=["#3a6ea5", "#c1432d"])
    ax.set_title("Distribución de la variable objetivo (Exited)")
    ax.set_xlabel("Exited"); ax.set_ylabel("Frecuencia")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_target.png", dpi=140)
    plt.close(fig)

    # Histogramas de variables numéricas separados por clase.
    num = df.select_dtypes(include=np.number).columns.difference([ID_COL, TARGET])
    fig, axes = plt.subplots(int(np.ceil(len(num) / 3)), 3, figsize=(13, 3 * np.ceil(len(num) / 3)))
    for ax, col in zip(axes.ravel(), num):
        sns.histplot(data=df, x=col, hue=TARGET, kde=False, ax=ax, bins=30,
                     palette={0: "#3a6ea5", 1: "#c1432d"})
        ax.set_title(col)
    for ax in axes.ravel()[len(num):]:
        ax.set_visible(False)
    fig.suptitle("Distribución de variables numéricas por clase")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_numerics.png", dpi=140)
    plt.close(fig)

    # Categóricas vs target.
    cat = [c for c in df.select_dtypes(include="object").columns if c not in DROP_COLS]
    if cat:
        fig, axes = plt.subplots(1, len(cat), figsize=(5 * len(cat), 4))
        if len(cat) == 1:
            axes = [axes]
        for ax, col in zip(axes, cat):
            tab = pd.crosstab(df[col], df[TARGET], normalize="index")
            tab.plot(kind="bar", stacked=True, ax=ax, color=["#3a6ea5", "#c1432d"])
            ax.set_title(f"Tasa de fuga por {col}")
            ax.set_ylabel("Proporción")
        fig.tight_layout(); fig.savefig(FIG_DIR / "eda_categorical.png", dpi=140)
        plt.close(fig)

    # Matriz de correlación.
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.select_dtypes(include=np.number).drop(columns=[ID_COL]).corr()
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Correlaciones (Pearson)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_corr.png", dpi=140)
    plt.close(fig)

    with open(RES_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Figuras EDA guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
