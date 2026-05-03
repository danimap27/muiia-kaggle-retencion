#!/bin/bash
#SBATCH --job-name=kaggle-predict
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

# Modelo a usar: lgbm | xgb | cat | stacking. Por defecto lgbm.
MODEL=${1:-lgbm}

module load Miniconda3
source activate kaggle-retencion

cd "$SLURM_SUBMIT_DIR"
python -m src.predict --model "$MODEL" --out "submission_${MODEL}.csv"
