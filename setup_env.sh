#!/bin/bash
# ----------------------------------------------------------------------------
# Crea/actualiza el entorno conda 'kaggle-retencion' con todas las dependencias.
# Verifica imports al final y aborta si alguna librería falla.
#
# Uso (en sesión interactiva con internet):
#   salloc --mem=8G -c 4 -t 02:00:00 srun --pty /bin/bash -i
#   bash setup_env.sh
# ----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="kaggle-retencion"
PY_VERSION="3.11"

module load Miniconda3

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Entorno '${ENV_NAME}' ya existe."
else
    echo "[setup] Creando entorno '${ENV_NAME}' con Python ${PY_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PY_VERSION}" -y
fi

source activate "${ENV_NAME}"

echo "[setup] Pip install desde requirements.txt..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "[setup] Verificando imports críticos..."
python - <<'PY'
import importlib, sys
mods = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn",
    "xgboost", "lightgbm", "catboost", "optuna",
    "imblearn", "shap", "joblib", "yaml",
]
fail = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        fail.append(m)
if fail:
    sys.exit(f"[setup] Faltan: {fail}")
PY

echo "[setup] Listo. Activar con: 'module load Miniconda3 && source activate ${ENV_NAME}'"
