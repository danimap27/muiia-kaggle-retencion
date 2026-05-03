"""
Búsqueda de hiperparámetros con Optuna para los modelos más prometedores
(XGBoost, LightGBM, CatBoost, RandomForest, HistGB y MLP).

Cada estudio Optuna devuelve los mejores hiperparámetros junto con la
puntuación F1 promedio en CV estratificada de 5 folds. Los resultados se
guardan en results/optuna_<modelo>.json para reusarlos al entrenar el
modelo final.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from .config import TRAIN_CSV, RES_DIR, SEED
from .preprocessing import split_X_y, make_preprocessor
from .cv import stratified_kfold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)


def _score(estimator, X, y) -> float:
    """F1 medio en CV estratificada de 5 folds."""
    from sklearn.model_selection import cross_val_score
    return float(np.mean(cross_val_score(
        estimator, X, y, cv=stratified_kfold(), scoring="f1", n_jobs=1,
    )))


def _wrap(clf, scale: bool) -> Pipeline:
    return Pipeline(steps=[("preprocessor", make_preprocessor(scale=scale)),
                            ("classifier", clf)])


# ---------- Espacios de búsqueda ----------

def objective_xgb(trial, X, y):
    from xgboost import XGBClassifier
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    clf = XGBClassifier(
        **params, eval_metric="logloss", tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )
    return _score(_wrap(clf, scale=False), X, y)


def objective_lgbm(trial, X, y):
    from lightgbm import LGBMClassifier
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    clf = LGBMClassifier(
        **params, class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    return _score(_wrap(clf, scale=False), X, y)


def objective_cat(trial, X, y):
    from catboost import CatBoostClassifier
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    clf = CatBoostClassifier(
        **params, auto_class_weights="Balanced", verbose=0, random_seed=SEED,
    )
    return _score(_wrap(clf, scale=False), X, y)


def objective_rf(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
    }
    clf = RandomForestClassifier(
        **params, class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    return _score(_wrap(clf, scale=False), X, y)


def objective_histgb(trial, X, y):
    params = {
        "max_iter": trial.suggest_int("max_iter", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
    }
    clf = HistGradientBoostingClassifier(
        **params, class_weight="balanced", random_state=SEED,
    )
    return _score(_wrap(clf, scale=False), X, y)


def objective_mlp(trial, X, y):
    h1 = trial.suggest_int("h1", 32, 256, step=32)
    h2 = trial.suggest_int("h2", 16, 128, step=16)
    params = {
        "hidden_layer_sizes": (h1, h2),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
    }
    clf = MLPClassifier(
        **params, max_iter=500, random_state=SEED, early_stopping=True,
    )
    return _score(_wrap(clf, scale=True), X, y)


def objective_gb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
    }
    clf = GradientBoostingClassifier(**params, random_state=SEED)
    return _score(_wrap(clf, scale=False), X, y)


def objective_svc(trial, X, y):
    params = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
    }
    clf = SVC(**params, class_weight="balanced", probability=True, random_state=SEED)
    return _score(_wrap(clf, scale=True), X, y)


OBJECTIVES = {
    "xgb": objective_xgb,
    "lgbm": objective_lgbm,
    "cat": objective_cat,
    "rf": objective_rf,
    "histgb": objective_histgb,
    "mlp": objective_mlp,
    "gb": objective_gb,
    "svc": objective_svc,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(OBJECTIVES.keys()), required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Tiempo máximo en segundos (opcional)")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(df)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    import time
    t0 = time.time()
    def _cb(study, trial):
        elapsed = time.time() - t0
        remaining = elapsed / max(trial.number + 1, 1) * (args.n_trials - trial.number - 1)
        print(f"[trial {trial.number+1}/{args.n_trials}] "
              f"F1={trial.value:.4f} | best={study.best_value:.4f} | "
              f"elapsed={elapsed/60:.1f}min | ETA={remaining/60:.1f}min", flush=True)

    study.optimize(
        lambda t: OBJECTIVES[args.model](t, X, y),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
        callbacks=[_cb],
    )

    best = {"model": args.model, "best_value": study.best_value, "best_params": study.best_params,
            "n_trials": len(study.trials)}
    out_path = RES_DIR / f"optuna_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Modelo {args.model}: mejor F1={study.best_value:.4f}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
