import argparse
import os
import json
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    Trainer, TrainingArguments, BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
import torch
from peft import LoraConfig, get_peft_model
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
    # Set tokenizers parallelism explicitly
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
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

    # DDP setup if needed
    if args.ddp_worldsize > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.ddp_worldsize,
            rank=args.ddp_rank
        )
        local_rank = args.ddp_rank
    else:
        local_rank = 0

    # Load dataset
    dataset = load_dataset("Nikkozhang/LMTutor")['train']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        num_proc=4  # Added explicit num_proc
    )

    # Convert dataset to torch format
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Model configuration
    target_modules = [
        "model.layers.30.self_attn.q_proj",
        "model.layers.30.self_attn.k_proj",
        "model.layers.30.self_attn.v_proj",
        "model.layers.30.self_attn.o_proj",
        "model.layers.30.mlp.gate_proj",
        "model.layers.30.mlp.up_proj",
        "model.layers.30.mlp.down_proj",
        "model.layers.31.self_attn.q_proj",
        "model.layers.31.self_attn.k_proj",
        "model.layers.31.self_attn.v_proj",
        "model.layers.31.self_attn.o_proj",
        "model.layers.31.mlp.gate_proj",
        "model.layers.31.mlp.up_proj",
        "model.layers.31.mlp.down_proj"
    ]

    config = AutoConfig.from_pretrained(args.model)
    config.max_new_tokens = 800
    config.use_cache = False  # Important for training
    
    # Quantization and model setup
    if args.bnb4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    # Initialize the model with proper device mapping
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        quantization_config=quantization_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(base_model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=200,
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        local_rank=local_rank,
        fp16=True if torch.cuda.is_available() else False,
        remove_unused_columns=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Training
    trainer.train()

    # Save and push to hub
    trainer.save_model(args.output_dir)
    if local_rank == 0:  # Only push from main process
        trainer.push_to_hub()

    if args.ddp_worldsize > 1:
        torch.distributed.destroy_process_group()

    print("Model fine-tuned and saved successfully.")