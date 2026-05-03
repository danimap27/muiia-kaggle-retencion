"""
Resumen rápido del estado del proyecto: qué runs han terminado, cuáles
fallaron, métricas de baseline y mejor F1 conocido por modelo.

Uso:
    python -m src.status
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import RES_DIR


def main() -> None:
    print("\n" + "=" * 60)
    print("  Proyecto Kaggle Retención — Estado actual")
    print("=" * 60)

    # 1. Runs por fase.
    rows = []
    for d in sorted(RES_DIR.glob("*/run.csv")):
        try:
            r = pd.read_csv(d).iloc[0].to_dict()
            rows.append(r)
        except Exception:
            pass
    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby("phase").agg(
            total=("run_id", "count"),
            ok=("exit_code", lambda s: int((s == 0).sum())),
            fail=("exit_code", lambda s: int((s != 0).sum())),
            mean_sec=("elapsed_sec", "mean"),
        )
        print("\nRuns por fase:")
        print(summary.to_string())

    # 2. Baseline ranking.
    cv_path = RES_DIR / "models_cv.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path).sort_values("f1_mean", ascending=False)
        print(f"\nBaseline ({len(cv)} modelos), top 5 por F1:")
        cols = [c for c in ["model", "f1_mean", "f1_std", "roc_auc_mean", "wallclock"]
                if c in cv.columns]
        print(cv[cols].head(5).to_string(index=False))

    # 3. Mejores hiperparámetros encontrados.
    print("\nMejor F1 por Optuna:")
    for p in sorted(RES_DIR.glob("optuna_*.json")):
        try:
            j = json.load(open(p))
            print(f"  {j['model']:>8s}: F1={j['best_value']:.4f} ({j['n_trials']} trials)")
        except Exception:
            pass

    # 4. Umbrales óptimos.
    print("\nUmbrales óptimos (F1):")
    for p in sorted(RES_DIR.glob("threshold_*.json")):
        try:
            j = json.load(open(p))
            base = j.get("f1_at_0.5")
            base_s = f" (vs 0.5: {base:.4f})" if base else ""
            print(f"  {j['model']:>8s}: t={j['best_threshold']:.3f} | F1={j['best_f1']:.4f}{base_s}")
        except Exception:
            pass

    # 5. Submissions disponibles.
    subs = sorted(RES_DIR.glob("submission_*.csv"))
    if subs:
        print(f"\nSubmissions ({len(subs)}):")
        for s in subs:
            df_s = pd.read_csv(s)
            pos_rate = df_s["Exited"].mean()
            print(f"  {s.name}: n={len(df_s)} | pos_rate={pos_rate:.3f}")
    print()


if __name__ == "__main__":
    main()
