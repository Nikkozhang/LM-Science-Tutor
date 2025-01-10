#!/usr/bin/env bash

#SBATCH --job-name=test

#SBATCH --partition=gpu
#SBATCH -G 1

#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1
export HF_HOME=/share/data/mei-work/nikkozhang/LM-Science-Tutor/huggingface_cache
export HUGGINGFACE_TOKEN="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"

huggingface-cli login --token $HUGGINGFACE_TOKEN


srun python find_layer.py