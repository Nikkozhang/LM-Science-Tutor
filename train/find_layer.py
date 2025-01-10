import torch
from transformers import AutoModelForCausalLM

model_name = "princeton-nlp/Llemma-7B-32K-MathMix"
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)