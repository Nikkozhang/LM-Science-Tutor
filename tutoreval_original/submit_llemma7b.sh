#!/usr/bin/env bash

#SBATCH --job-name=generate-answers

#SBATCH --partition=gpu
#SBATCH -G 1

#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1
export HF_HOME=/share/data/mei-work/nikkozhang/LM-Science-Tutor/huggingface_cache

python -m generate --model princeton-nlp/Llemma-7B-32K-MathMix