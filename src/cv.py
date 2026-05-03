"""
Helpers de validación cruzada y métricas. Centralizar aquí evita que los
distintos scripts difieran en la forma de medir.
"""
from __future__ import annotations

from typing import Callable, Dict, List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, precision_score, recall_score,
)

from .config import N_SPLITS, SEED, RES_DIR


def stratified_kfold():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


SCORERS = {
    "f1": "f1",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def cv_evaluate(estimator, X, y, name: str) -> Dict:
    """Ejecuta CV estratificado y devuelve un dict con la media y desviación
    de cada métrica, además de los valores por fold."""
    skf = stratified_kfold()
    res = cross_validate(estimator, X, y, cv=skf, scoring=SCORERS,
                         n_jobs=1, return_train_score=False)
    out = {"model": name}
    for k in SCORERS:
        v = res[f"test_{k}"]
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
        out[f"{k}_folds"] = [float(x) for x in v]
    out["fit_time"] = float(np.mean(res["fit_time"]))
    return out


def append_results(records: List[Dict], path: str = "models_cv.csv") -> None:
    """Concatena con resultados previos (si los hay) y reescribe ordenado por F1.

    Si no hay records nuevos no se hace nada (evita romper cuando se llama
    con un modelo no disponible).
    """
    if not records:
        print("Sin resultados para registrar.")
        return

    out_path = RES_DIR / path
    if out_path.exists():
        prev = pd.read_csv(out_path).to_dict("records")
        # Sustituye entradas con el mismo nombre de modelo si vuelven a aparecer.
        names_new = {r["model"] for r in records}
        prev = [r for r in prev if r.get("model") not in names_new]
        records = prev + records

    df = pd.DataFrame(records)
    if "f1_mean" in df.columns:
        df = df.sort_values("f1_mean", ascending=False)
    df.to_csv(out_path, index=False)
    with open(RES_DIR / path.replace(".csv", ".json"), "w") as f:
        json.dump(records, f, indent=2)
