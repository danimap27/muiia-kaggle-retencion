#!/bin/bash
#SBATCH --job-name=kaggle-baseline
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

module load Miniconda3
source activate kaggle-retencion

cd "$SLURM_SUBMIT_DIR"
python -m src.train_baseline
