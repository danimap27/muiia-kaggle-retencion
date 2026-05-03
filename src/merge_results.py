"""
Consolida los resultados parciales de array jobs SLURM.

Cada tarea de baseline escribe ``results/models_cv.csv`` con uno o varios
modelos. Cuando se ejecutan en paralelo pueden pisarse. Este script lee
todos los ``stdout.log`` de las carpetas ``results/baseline__*/`` para
extraer las métricas finales y rehace ``models_cv.csv`` desde cero.

Si por algún motivo no encuentra métricas en los logs intenta usar el
fichero ``models_cv.json`` actual como fuente.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import RES_DIR


_METRICS_RE = re.compile(
    r"\[\d+/\d+\]\s+(\S+)\s+\|\s+F1=([\d.]+)\s+±\s+([\d.]+)\s+\|\s+AUC=([\d.]+)\s+\|\s+t=([\d.]+)s"
)


def _parse_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = _METRICS_RE.search(text)
    if not m:
        return None
    return {
        "model": m.group(1),
        "f1_mean": float(m.group(2)),
        "f1_std": float(m.group(3)),
        "roc_auc_mean": float(m.group(4)),
        "wallclock": float(m.group(5)),
    }


def main() -> None:
    rows: list[dict] = []
    # Recoge métricas desde los logs de cada tarea baseline.
    for d in sorted(RES_DIR.glob("baseline__*/")):
        rec = _parse_log(d / "stdout.log")
        if rec:
            rec["run_dir"] = d.name
            rows.append(rec)

    # Si no se ha podido extraer nada, conserva el JSON anterior.
    if not rows and (RES_DIR / "models_cv.json").exists():
        rows = json.load(open(RES_DIR / "models_cv.json"))

    if not rows:
        print("No se ha encontrado ninguna métrica de baseline.")
        return

    df = pd.DataFrame(rows).sort_values("f1_mean", ascending=False)
    df.to_csv(RES_DIR / "models_cv.csv", index=False)
    with open(RES_DIR / "models_cv.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"{len(df)} modelos consolidados en results/models_cv.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
