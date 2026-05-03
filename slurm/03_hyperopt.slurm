#!/bin/bash
#SBATCH --job-name=kaggle-hyperopt
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hyperopt_%A_%a.out
#SBATCH --error=logs/hyperopt_%A_%a.err
#SBATCH --array=0-5

# Job array: cada índice optimiza un modelo distinto en paralelo.
MODELS=(xgb lgbm cat rf histgb mlp)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

module load Miniconda3
source activate kaggle-retencion

cd "$SLURM_SUBMIT_DIR"
python -m src.hyperopt --model "$MODEL" --n-trials 80 --timeout 36000
