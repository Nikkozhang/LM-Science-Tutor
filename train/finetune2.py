import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def setup(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix", type=str)
    parser.add_argument("--output_dir", default="./fine-tuned-model-fsdp", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--hub_model_id", default="Nikkozhang/LoraTuned_1", type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)

    args = parser.parse_args()
    world_size = args.world_size
    rank = args.rank

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    setup(rank, world_size)

    # Load dataset
    dataset = load_dataset("Nikkozhang/LMTutor")['train']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Tokenize dataset
    def tokenize_function(examples):
        input_texts = [output + " " + question for output, question in zip(examples['output'], examples['original_question'])]
        inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=512)
        inputs["labels"] = targets["input_ids"].copy()
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['output', 'original_question', 'response'])
    
    # Creating DataLoader
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Setup FSDP
    if world_size > 1:
        auto_wrap_policy = size_based_auto_wrap_policy(10**8)  # Adjust the threshold as needed
        fsdp_model = FSDP(model, auto_wrap_policy=auto_wrap_policy).to(rank)
    else:
        fsdp_model = model.to(rank)

    # Training arguments
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
        push_to_hub=True,
        hub_model_id=args.hub_model_id
    )

    # Trainer
    trainer = Trainer(
        model=fsdp_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None  # You could set evaluation dataset here if applicable
    )

    # Train model
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()

    cleanup()
    print("Model fine-tuned and saved successfully.")