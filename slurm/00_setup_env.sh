#!/bin/bash
# ----------------------------------------------------------------------------
# Crea el entorno conda 'kaggle-retencion' e instala dependencias con pip.
# Se ejecuta UNA vez en sesión interactiva (necesita internet).
#
# Uso:
#   salloc --mem=8G -c 4 -t 02:00:00 srun --pty /bin/bash -i
#   bash slurm/00_setup_env.sh
# ----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="kaggle-retencion"
PY_VERSION="3.11"

module load Miniconda3

CONDA_BASE="/lustre/software/easybuild/common/software/Miniconda3/4.9.2"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Entorno '${ENV_NAME}' ya existe; se omite la creación."
else
    echo "Creando entorno '${ENV_NAME}' con Python ${PY_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PY_VERSION}" -y
fi

source activate "${ENV_NAME}"

echo "Instalando dependencias (pip)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verificando entorno..."
python -c "import sklearn, xgboost, lightgbm, catboost, optuna; print('Entorno OK')"
echo "Listo. Para activar: 'module load Miniconda3 && source activate ${ENV_NAME}'"
