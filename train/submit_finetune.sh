#!/usr/bin/env bash

#SBATCH --job-name=finetune
#SBATCH --partition=mm-gpu
#SBATCH --ntasks=8         # Total number of tasks
#SBATCH --nodes=1          # Request 1 node
#SBATCH --gpus=8          # Request 8 GPUs in total
#SBATCH --cpus-per-task=1  # Allocate CPU cores per task
#SBATCH --output=slurm_%A.out  # Output file
#SBATCH --error=slurm_%A.err   # Error file
#SBATCH --open-mode=append
#SBATCH --export=ALL,IS_REMOTE=1

# Set environment variables
export HF_HOME=/share/data/mei-work/nikkozhang/LM-Science-Tutor/huggingface_cache
export HUGGINGFACE_TOKEN="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"
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)

# Login to Hugging Face
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Parameters
MODEL="princeton-nlp/Llemma-7B-32K-MathMix"
OUTPUT_DIR="./fine-tuned-model-lora"
BATCH_SIZE=2
NUM_TRAIN_EPOCHS=3
HUB_MODEL_ID="Nikkozhang/LoraTuned_1"

# Run the Python script with DDP
srun \
    --exclusive \
    --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 \
    python -u finetune.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --hub_model_id $HUB_MODEL_ID \
    --ddp_worldsize 8 \
    --bnb4bit