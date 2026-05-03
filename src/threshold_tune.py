"""
Optimiza el umbral de decisión para maximizar F1 sobre validación cruzada.

Por defecto los clasificadores usan 0.5 como corte. Con clases desbalanceadas
y métrica F1, el óptimo está casi siempre por debajo de 0.5. Este script
busca el umbral que maximiza F1 en CV out-of-fold y lo guarda para usarlo
luego en predict.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_recall_curve

from .config import TRAIN_CSV, RES_DIR, SEED
from .preprocessing import split_X_y
from .cv import stratified_kfold
from .predict import build_final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm",
                        choices=["lgbm", "xgb", "cat", "histgb", "rf", "gb",
                                  "stacking", "blend"])
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(df)

    if args.model == "blend":
        # El umbral del blend se obtiene en src.blend (ya consolidado).
        bj = json.load(open(RES_DIR / "blend.json"))
        out = {"model": "blend", "best_threshold": bj["threshold"],
                "best_f1": bj["best_f1"], "f1_at_0.5": None}
        with open(RES_DIR / "threshold_blend.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"Blend threshold: {bj['threshold']:.3f} | F1={bj['best_f1']:.4f}")
        return

    if args.model == "rf":
        from .models import get_models
        pipe = get_models()["RandomForest"]
    elif args.model == "gb":
        from .models import get_models
        pipe = get_models()["GradBoost"]
    elif args.model == "histgb":
        from .models import get_models
        pipe = get_models()["HistGB"]
    else:
        pipe = build_final(args.model)

    print(f"Calculando probabilidades OOF para {args.model}...")
    proba = cross_val_predict(pipe, X, y, cv=stratified_kfold(),
                              method="predict_proba", n_jobs=1)[:, 1]

    # Búsqueda fina del umbral.
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    out = {
        "model": args.model,
        "best_threshold": best_t,
        "best_f1": best_f1,
        "f1_at_0.5": float(f1_score(y, (proba >= 0.5).astype(int))),
    }
    path = RES_DIR / f"threshold_{args.model}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Mejor umbral: {best_t:.3f} | F1={best_f1:.4f} (vs 0.5: {out['f1_at_0.5']:.4f})")
    print(f"Guardado en {path}")


if __name__ == "__main__":
    main()
