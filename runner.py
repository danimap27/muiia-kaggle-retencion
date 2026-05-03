"""
runner.py — adaptador hercules-framework para el proyecto Kaggle.

Cada fase definida en config.yaml se traduce en un fichero ``cmds_*.txt``
con una orden por línea. El array job de SLURM ejecuta cada línea como una
tarea independiente. La ejecución concreta de cada orden es un ``python -m``
sobre uno de los módulos de ``src/``.

Adaptación del template hercules-framework:
- ``iter_runs`` recorre las fases definidas y emite ``run_spec`` (dict con
  ``run_id`` y ``cmd`` ya construido).
- ``execute_run`` ejecuta el comando con ``subprocess`` y registra
  resultados en ``results/<run_id>/run.csv``.
"""
from __future__ import annotations

import argparse
import csv
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# SECTION 1 — Definición de runs por fase
# =============================================================================

def _runs_eda(cfg: dict) -> Iterable[dict]:
    yield {
        "run_id": "eda",
        "phase": "eda",
        "cmd": "python -m src.eda",
    }


def _runs_baseline(cfg: dict) -> Iterable[dict]:
    """Una tarea por modelo del catálogo (paraleliza en SLURM array)."""
    for m in cfg["baseline_models"]:
        yield {
            "run_id": f"baseline__{m}",
            "phase": "baseline",
            "model": m,
            "cmd": f"python -m src.train_baseline --only {m}",
        }


def _runs_hyperopt(cfg: dict) -> Iterable[dict]:
    for entry in cfg["hyperopt_models"]:
        m = entry["name"]
        n = entry["trials"]
        yield {
            "run_id": f"hyperopt__{m}",
            "phase": "hyperopt",
            "model": m,
            "trials": n,
            "cmd": f"python -m src.hyperopt --model {m} --n-trials {n}",
        }


def _runs_threshold(cfg: dict) -> Iterable[dict]:
    for m in cfg["threshold_models"]:
        yield {
            "run_id": f"threshold__{m}",
            "phase": "threshold",
            "model": m,
            "cmd": f"python -m src.threshold_tune --model {m}",
        }


def _runs_stacking(cfg: dict) -> Iterable[dict]:
    yield {
        "run_id": "stacking",
        "phase": "stacking",
        "cmd": "python -m src.stacking",
    }


def _runs_predict(cfg: dict) -> Iterable[dict]:
    for m in cfg["predict_models"]:
        yield {
            "run_id": f"predict__{m}",
            "phase": "predict",
            "model": m,
            "cmd": f"python -m src.predict --model {m} --out submission_{m}.csv",
        }


PHASE_BUILDERS = {
    "eda": _runs_eda,
    "baseline": _runs_baseline,
    "hyperopt": _runs_hyperopt,
    "threshold": _runs_threshold,
    "stacking": _runs_stacking,
    "predict": _runs_predict,
}


def iter_runs(cfg: dict) -> list[dict]:
    """Devuelve todas las runs (todas las fases concatenadas)."""
    runs: list[dict] = []
    for phase in cfg["phases"]:
        kind = phase.get("kind", phase["name"])
        builder = PHASE_BUILDERS.get(kind)
        if builder is None:
            logger.warning(f"Fase desconocida: {kind}")
            continue
        for r in builder(cfg):
            runs.append(r)
    return runs


# =============================================================================
# SECTION 2 — Ejecución de un run individual
# =============================================================================

def execute_run(run_spec: dict, cfg: dict, machine_id: str = "local") -> int:
    run_id = run_spec["run_id"]
    cmd = run_spec["cmd"]

    out_root = Path(cfg.get("output_dir", "./results"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "run.csv"

    logger.info(f"[START] {run_id} | {cmd}")
    t0 = time.time()
    log_path = run_dir / "stdout.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run(shlex.split(cmd), stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    row = {
        "run_id": run_id,
        "phase": run_spec.get("phase", ""),
        "model": run_spec.get("model", ""),
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "machine_id": machine_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": cmd,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)

    status = "DONE" if proc.returncode == 0 else "FAIL"
    logger.info(f"[{status}] {run_id} | exit={proc.returncode} | t={elapsed:.1f}s")
    return proc.returncode


# =============================================================================
# SECTION 3 — Filtrado por fase para los .txt de comandos
# =============================================================================

def runs_of_phase(all_runs: list[dict], phase: dict) -> list[dict]:
    kind = phase.get("kind", phase["name"])
    return [r for r in all_runs if r.get("phase") == kind]


def export_commands(runs: list[dict], out_path: str, config_path: str) -> None:
    """Exporta una orden por línea para que SLURM la consuma."""
    lines = [f"python runner.py --config {config_path} --run-id {r['run_id']}"
             for r in runs]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Exported {len(lines)} commands -> {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export-commands", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--machine-id", default="local")
    args = ap.parse_args()

    cfg = load_config(args.config)
    all_runs = iter_runs(cfg)

    if args.export_commands:
        for phase in cfg["phases"]:
            filtered = runs_of_phase(all_runs, phase)
            export_commands(filtered, phase["file"], args.config)
        return

    if args.count:
        from collections import Counter
        c = Counter(r["phase"] for r in all_runs)
        print(f"Total runs: {len(all_runs)}")
        for k, v in c.items():
            print(f"  {k}: {v}")
        return

    if args.run_id:
        spec = next((r for r in all_runs if r["run_id"] == args.run_id), None)
        if spec is None:
            logger.error(f"run_id {args.run_id!r} no encontrado")
            sys.exit(2)
        sys.exit(execute_run(spec, cfg, args.machine_id))

    if args.dry_run:
        logger.info(f"Planned runs: {len(all_runs)}")
        for r in all_runs:
            print(f"  {r['run_id']:30s} | {r['cmd']}")
        return

    # Sin run-id: ejecuta todo en serie (uso local).
    rc = 0
    for r in all_runs:
        rc |= execute_run(r, cfg, args.machine_id)
    sys.exit(rc)


if __name__ == "__main__":
    main()
