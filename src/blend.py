"""
Blending probabilístico de los modelos optimizados con Optuna.

Para cada modelo de la lista (cat, histgb, xgb, lgbm) se obtienen las
probabilidades out-of-fold mediante CV estratificada y se busca el
conjunto de pesos no negativos sumando 1 que maximiza el F1 (con
optimización del umbral conjunta). El resultado se guarda para la
fase de predicción.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

from .config import TRAIN_CSV, RES_DIR, SEED
from .preprocessing import split_X_y
from .cv import stratified_kfold
from .predict import build_final


def _oof_proba(model: str, X, y) -> np.ndarray:
    """Devuelve probabilidades OOF (out-of-fold) para un modelo."""
    pipe = build_final(model)
    return cross_val_predict(pipe, X, y, cv=stratified_kfold(),
                              method="predict_proba", n_jobs=1)[:, 1]


def _f1_blend(weights: np.ndarray, probas: np.ndarray, y, threshold: float) -> float:
    p = probas @ weights
    return f1_score(y, (p >= threshold).astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    blend_models = cfg.get("blend_models", ["cat", "histgb", "xgb", "lgbm"])

    df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(df)

    print(f"Calculando probabilidades OOF para: {blend_models}")
    proba_matrix: List[np.ndarray] = []
    used: List[str] = []
    for m in blend_models:
        try:
            p = _oof_proba(m, X, y)
            proba_matrix.append(p)
            used.append(m)
            f1_solo = f1_score(y, (p >= 0.5).astype(int))
            print(f"  {m}: F1@0.5={f1_solo:.4f}")
        except Exception as e:
            print(f"  {m}: SKIP ({e})")
    if not proba_matrix:
        raise SystemExit("No hay modelos disponibles para el blend.")

    P = np.column_stack(proba_matrix)
    n = P.shape[1]

    # Búsqueda conjunta de pesos y umbral.
    best = {"f1": 0.0, "weights": None, "threshold": 0.5}
    for thr in np.linspace(0.2, 0.7, 26):
        # Inicialización uniforme.
        w0 = np.ones(n) / n

        def neg_f1(w):
            w = np.clip(w, 0, None)
            s = w.sum()
            if s == 0:
                return 0.0
            w = w / s
            return -_f1_blend(w, P, y, thr)

        res = minimize(neg_f1, w0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 200})
        w = np.clip(res.x, 0, None)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        f1_now = -res.fun
        if f1_now > best["f1"]:
            best = {"f1": float(f1_now), "weights": w.tolist(),
                    "threshold": float(thr)}

    # Resultado.
    out = {
        "models": used,
        "weights": best["weights"],
        "threshold": best["threshold"],
        "best_f1": best["f1"],
    }
    out_path = RES_DIR / "blend.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBlend óptimo: F1={best['f1']:.4f} | thr={best['threshold']:.3f}")
    for m, w in zip(used, best["weights"]):
        print(f"  {m:>8s}: {w:.3f}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
