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
export HUGGINGFACE_TOKEN="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"

# Add token to Hugging Face environment
huggingface-cli login --token $HUGGINGFACE_TOKEN

srun vllm serve "deepseek-ai/DeepSeek-V2" --port 8000 --trust-remote-code &

# 2. Give time for the server to load
sleep 30

# 3. Run your Python script (will POST to http://localhost:8000).
srun --exclusive --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 python generate_deepseek.py --model deepseek-ai/DeepSeek-V2 --closedbook --ddp_worldsize 8 --ddp_rank $SLURM_PROCID --vllm