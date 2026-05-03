#!/bin/bash
# ----------------------------------------------------------------------------
# Crea el entorno conda 'kaggle-retencion' a partir de environment.yml.
# Se ejecuta UNA vez en una sesión interactiva (no como job batch porque
# requiere descargas de internet).
#
# Uso:
#   salloc --mem=8G -c 4 -t 02:00:00 srun --pty /bin/bash -i
#   bash slurm/00_setup_env.sh
# ----------------------------------------------------------------------------
set -euo pipefail

module load Miniconda3

if conda env list | grep -q "kaggle-retencion"; then
    echo "El entorno 'kaggle-retencion' ya existe; se omite la creación."
else
    conda env create -f environment.yml
fi

source activate kaggle-retencion
python -c "import sklearn, xgboost, lightgbm, catboost, optuna; print('Entorno OK')"
