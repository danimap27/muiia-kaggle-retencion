"""
Reentrena el modelo final con todos los datos de train, predice sobre test
y genera el submission.csv con el formato CustomerId,Exited.

Se puede usar el modelo individual ganador (lightgbm, xgb, etc.) o el
stacking. Por defecto: lightgbm con los mejores hiperparámetros de Optuna.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import TRAIN_CSV, TEST_CSV, RES_DIR, MODELS_DIR, ID_COL, TARGET, SEED
from .preprocessing import split_X_y, make_preprocessor
from sklearn.pipeline import Pipeline


def _load_params(name: str):
    p = RES_DIR / f"optuna_{name}.json"
    if p.exists():
        return json.load(open(p))["best_params"]
    return {}


def build_final(model: str) -> Pipeline:
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            **_load_params("lgbm"),
            class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
        )
        scale = False
    elif model == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            **_load_params("xgb"),
            eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1,
        )
        scale = False
    elif model == "cat":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            **_load_params("cat"),
            auto_class_weights="Balanced", verbose=0, random_seed=SEED,
        )
        scale = False
    elif model == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            **_load_params("histgb"),
            class_weight="balanced", random_state=SEED,
        )
        scale = False
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            **_load_params("rf"),
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        )
        scale = False
    elif model == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(**_load_params("gb"), random_state=SEED)
        scale = False
    elif model == "mlp":
        from sklearn.neural_network import MLPClassifier
        params = _load_params("mlp")
        if "h1" in params:
            params["hidden_layer_sizes"] = (params.pop("h1"), params.pop("h2"))
        clf = MLPClassifier(
            **params, max_iter=500, random_state=SEED, early_stopping=True,
        )
        scale = True
    elif model == "svc":
        from sklearn.svm import SVC
        clf = SVC(
            **_load_params("svc"), probability=True,
            class_weight="balanced", random_state=SEED,
        )
        scale = True
    elif model == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs",
            random_state=SEED,
        )
        scale = True
    elif model == "stacking":
        return joblib.load(MODELS_DIR / "stacking.joblib")
    else:
        raise ValueError(f"Modelo desconocido: {model}")

    return Pipeline(steps=[("preprocessor", make_preprocessor(scale=scale)),
                            ("classifier", clf)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm",
                        choices=["lgbm", "xgb", "cat", "histgb", "rf", "gb",
                                  "mlp", "svc", "logreg", "stacking", "blend"])
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    X_train, y_train = split_X_y(train)
    X_test, _ = split_X_y(test)

    if args.model == "blend":
        # Blend de probabilidades de modelos individuales reentrenados.
        blend = json.load(open(RES_DIR / "blend.json"))
        print(f"Blend: pesos {dict(zip(blend['models'], blend['weights']))}")
        probas = np.zeros(len(X_test))
        for m, w in zip(blend["models"], blend["weights"]):
            sub_pipe = build_final(m)
            sub_pipe.fit(X_train, y_train)
            probas += w * sub_pipe.predict_proba(X_test)[:, 1]
        thr = blend["threshold"]
        preds = (probas >= thr).astype(int)
        print(f"Aplicado umbral del blend: {thr:.3f}")
    else:
        pipe = build_final(args.model)
        if args.model != "stacking":
            print(f"Reentrenando {args.model} sobre {len(X_train)} muestras...")
            pipe.fit(X_train, y_train)
            joblib.dump(pipe, MODELS_DIR / f"{args.model}_final.joblib")

        print(f"Generando predicciones sobre {len(X_test)} muestras...")
        thr_path = RES_DIR / f"threshold_{args.model}.json"
        if thr_path.exists() and hasattr(pipe, "predict_proba"):
            thr = json.load(open(thr_path))["best_threshold"]
            proba = pipe.predict_proba(X_test)[:, 1]
            preds = (proba >= thr).astype(int)
            print(f"Aplicado umbral optimizado: {thr:.3f}")
        else:
            preds = pipe.predict(X_test).astype(int)

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: preds})
    out_path = RES_DIR / args.out
    sub.to_csv(out_path, index=False)
    print(f"Submission guardada en {out_path} ({sub[TARGET].mean():.3f} positivos)")


if __name__ == "__main__":
    main()
