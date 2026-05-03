"""
Evaluación con CV de todos los modelos del catálogo.

No hay búsqueda de hiperparámetros aquí: los hiperparámetros son los del
docstring de cada modelo en ``models.py``. La idea es tener una foto rápida
del rendimiento relativo antes de invertir tiempo en optimizar los buenos.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .config import TRAIN_CSV, RES_DIR
from .preprocessing import split_X_y
from .models import get_models
from .cv import cv_evaluate, append_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Lista de modelos a evaluar (por defecto todos)")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(df)

    catalog = get_models()
    if args.only:
        catalog = {k: v for k, v in catalog.items() if k in args.only}

    print(f"Evaluando {len(catalog)} modelos con CV estratificada (5 folds)")
    records = []
    for name, pipe in catalog.items():
        t0 = time.time()
        try:
            rec = cv_evaluate(pipe, X, y, name)
            rec["wallclock"] = round(time.time() - t0, 1)
            records.append(rec)
            print(f"{name:>14s} | F1={rec['f1_mean']:.4f} ± {rec['f1_std']:.4f} | "
                  f"AUC={rec['roc_auc_mean']:.4f} | t={rec['wallclock']}s")
        except Exception as e:
            print(f"{name:>14s} | ERROR: {e}")
    append_results(records, "models_cv.csv")
    print(f"Resultados guardados en {RES_DIR/'models_cv.csv'}")


if __name__ == "__main__":
    main()
