#!/bin/bash
#SBATCH --job-name=kaggle-stacking
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stacking_%j.out
#SBATCH --error=logs/stacking_%j.err

module load Miniconda3
source activate kaggle-retencion

cd "$SLURM_SUBMIT_DIR"
python -m src.stacking
