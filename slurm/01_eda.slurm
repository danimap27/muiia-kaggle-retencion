#!/bin/bash
#SBATCH --job-name=kaggle-eda
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eda_%j.out
#SBATCH --error=logs/eda_%j.err

module load Miniconda3
source activate kaggle-retencion

cd "$SLURM_SUBMIT_DIR"
python -m src.eda
