import argparse
import os
import json
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    Trainer, TrainingArguments, BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
import torch
from peft import LoraConfig, LoraModel
from huggingface_hub import HfApi

def tokenize_function(examples):
    # Concatenate 'output' and 'original_question' for tokenization
    input_texts = [output + " " + question for output, question in zip(examples['output'], examples['original_question'])]
    
    # Tokenize the concatenated inputs
    inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=512)
    
    # Tokenize the targets (correct responses)
    targets = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=512)
    
    # Copy the input_ids from targets to be used as labels
    inputs["labels"] = targets["input_ids"].copy()
    
    return inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix", type=str, help="Generator model")
    parser.add_argument("--output_dir", default="./fine-tuned-model-lora", type=str, help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size used during training")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--bnb4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--hub_model_id", default="your-username/your-model-name", type=str, help="Model name for Hugging Face hub")
    parser.add_argument("--ddp_worldsize", default=1, type=int, help="Number of parallel instances for data parallelism")
    parser.add_argument("--ddp_rank", default=0, type=int, help="Rank of the data fragment for generation")

    args = parser.parse_args()

    if args.ddp_worldsize > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.ddp_worldsize,
            rank=args.ddp_rank
        )
        torch.cuda.set_device(args.ddp_rank)
        args.device = torch.device(f'cuda:{args.ddp_rank}')

    # Load dataset
    dataset = load_dataset("Nikkozhang/LMTutor")['train']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['output', 'original_question', 'response'])
    train_dataset = torch.utils.data.DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    # LoRA Configuration and Model Initialization
    target_modules = [
        "transformer.h.30.self_attn.q_proj", 
        "transformer.h.30.self_attn.k_proj", 
        "transformer.h.30.self_attn.v_proj", 
        "transformer.h.30.self_attn.o_proj", 
        "transformer.h.30.mlp.gate_proj", 
        "transformer.h.30.mlp.up_proj", 
        "transformer.h.30.mlp.down_proj", 
        "transformer.h.31.self_attn.q_proj", 
        "transformer.h.31.self_attn.k_proj", 
        "transformer.h.31.self_attn.v_proj", 
        "transformer.h.31.self_attn.o_proj", 
        "transformer.h.31.mlp.gate_proj", 
        "transformer.h.31.mlp.up_proj", 
        "transformer.h.31.mlp.down_proj"
    ]

    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=target_modules, 
        lora_dropout=0.1,
        bias="none"
    )

    config = AutoConfig.from_pretrained(args.model)
    config.max_new_tokens = 800
    config.dtype = torch.bfloat16
    config.do_sample = False
    config.use_cache = True
    
    if args.bnb4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    model = LoraModel(base_model=base_model, lora_config=lora_config)
    
    # Fine-tuning section
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=200,
        push_to_hub=True,  # Add this to enable pushing to the Hugging Face hub
        hub_model_id=args.hub_model_id,
        auto_wrap_policy=torch.distributed.fsdp.auto_wrap_policy  # Ensure coherence with DDP and auto wrapping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None
    )

    trainer.train()

    # Save and push the fine-tuned model to the Hugging Face hub
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()

    if args.ddp_worldsize > 1:
        torch.distributed.destroy_process_group()

    print("Model fine-tuned and saved successfully.")