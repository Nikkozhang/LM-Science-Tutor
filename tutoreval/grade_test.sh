#!/usr/bin/env bash

#SBATCH --job-name=grade_test

#SBATCH --partition=gpu
#SBATCH -G 1

#SBATCH --output=grade.out
#SBATCH --error=grade.err
#SBATCH --open-mode=append

#SBATCH --export=ALL,IS_REMOTE=1

if [ -z "$USER" ]; then
    export USER=$(whoami)
fi

export OPENAI_API_KEY="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"

python -m grade --input_file "tutoreval/generations/openbook/princeton-nlp/Llemma-7B-32K-MathMix_5A2.json"
