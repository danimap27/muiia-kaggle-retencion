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
    """Stacking con base diversa (gbms + mlp + svc) y meta-modelo LightGBM."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    estimators = [
        ("xgb", _xgb(_load_best("xgb"))),
        ("lgbm", _lgbm(_load_best("lgbm"))),
        ("cat", _cat(_load_best("cat"))),
        ("histgb", HistGradientBoostingClassifier(
            random_state=SEED, class_weight="balanced",
            **(_load_best("histgb") or {}))),
        ("rf", RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                              random_state=SEED, early_stopping=True)),
        ("svc", SVC(C=1.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=SEED)),
    ]
    final = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
    )
    stack = StackingClassifier(
        estimators=estimators, final_estimator=final,
        cv=5, passthrough=True, n_jobs=1,
    )
    return Pipeline(steps=[("preprocessor", make_preprocessor(scale=True)),
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
