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

# Parameters
MODEL="princeton-nlp/Llemma-7B-32K-MathMix"
OUTPUT_DIR="./fine-tuned-model-lora"
BATCH_SIZE=2
NUM_TRAIN_EPOCHS=3
HUB_MODEL_ID="Nikkozhang/LoraTuned_1"

# Run the Python script with the specified parameters
srun python finetune.py --model $MODEL --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --num_train_epochs $NUM_TRAIN_EPOCHS --hub_model_id $HUB_MODEL_ID