#!/usr/bin/env bash

#SBATCH --job-name=generate-answers
#SBATCH --partition=mm-gpu
#SBATCH --ntasks=8        # Total number of tasks
#SBATCH --nodes=1         # Request 1 node
#SBATCH --gpus=8          # Request 8 GPUs in total
#SBATCH --cpus-per-task=1 # Allocate CPU cores per task if needed
#SBATCH --output=slurm_%A.out  # Output file
#SBATCH --error=slurm_%A.err   # Error file
#SBATCH --open-mode=append
#SBATCH --export=ALL,IS_REMOTE=1

export HF_HOME=/share/data/mei-work/nikkozhang/LM-Science-Tutor/huggingface_cache

# Run the script
srun --exclusive --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 python generate_save_every1000.py --model google/gemma-7b --closedbook --ddp_worldsize 8 --ddp_rank $SLURM_PROCID --vllm
