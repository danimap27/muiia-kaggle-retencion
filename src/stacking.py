"""
Modelo de stacking. Combina predicciones de los mejores boostings y un
modelo lineal (logística regularizada) como meta-aprendiz.

El stacking se entrena con CV interna (passthrough=False) y se evalúa de
nuevo con CV externa para tener una estimación honesta del rendimiento.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import TRAIN_CSV, RES_DIR, MODELS_DIR, SEED
from .preprocessing import split_X_y, make_preprocessor
from .cv import cv_evaluate
import joblib


def _load_best(name: str):
    path = RES_DIR / f"optuna_{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["best_params"]


def _xgb(params=None):
    from xgboost import XGBClassifier
    p = params or dict(n_estimators=600, max_depth=5, learning_rate=0.05)
    return XGBClassifier(
        **p, eval_metric="logloss", tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )


def _lgbm(params=None):
    from lightgbm import LGBMClassifier
    p = params or dict(n_estimators=800, num_leaves=63, learning_rate=0.05)
    return LGBMClassifier(
        **p, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
    )


def _cat(params=None):
    from catboost import CatBoostClassifier
    p = params or dict(iterations=800, depth=6, learning_rate=0.05)
    return CatBoostClassifier(
        **p, auto_class_weights="Balanced", verbose=0, random_seed=SEED,
    )


def build_stack() -> Pipeline:
    """Construye el StackingClassifier completo dentro de un pipeline."""
    estimators = [
        ("xgb", _xgb(_load_best("xgb"))),
        ("lgbm", _lgbm(_load_best("lgbm"))),
        ("cat", _cat(_load_best("cat"))),
        ("rf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )),
    ]
    final = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    stack = StackingClassifier(
        estimators=estimators, final_estimator=final,
        cv=5, passthrough=False, n_jobs=1,
    )
    return Pipeline(steps=[("preprocessor", make_preprocessor(scale=False)),
                            ("classifier", stack)])


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(df)

    pipe = build_stack()
    print("Evaluando stacking con CV externa (5 folds)...")
    rec = cv_evaluate(pipe, X, y, "Stacking")
    print(f"Stacking: F1={rec['f1_mean']:.4f} ± {rec['f1_std']:.4f} | AUC={rec['roc_auc_mean']:.4f}")

    pipe.fit(X, y)
    joblib.dump(pipe, MODELS_DIR / "stacking.joblib")

    with open(RES_DIR / "stacking_cv.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Modelo guardado en {MODELS_DIR/'stacking.joblib'}")


if __name__ == "__main__":
    main()
