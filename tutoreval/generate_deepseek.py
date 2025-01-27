import argparse
import os
import json
from tqdm import tqdm
import requests  # Added for making HTTP requests to the vLLM server
from utils.openai_utils import OpenAI
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from utils.generation_utils import generation_utils
from datasets import load_dataset
from datasets import Dataset
import torch
# Removed vllm imports since we're using the server instead
# from vllm import LLM, SamplingParams

class RequestOutput(dict):
    def __init__(self, data):
        super().__init__(data=data)

def save_outputs(outputs, save_dir, batch_index, rank):
    save_path = os.path.join(save_dir, f"output_rank{rank}_batch{batch_index}.json")
    with open(save_path, "a") as f:
        json.dump(outputs, f, indent=4)
        f.write('\n')

def generate_answers(data, template, model, tokenizer=None, server_url=None, model_name=None):
    outputs = []
    batch_index = 0
    for sample in tqdm(data):
        questions = sample["query"]
        sample["template"] = [template] * len(questions)
        query = [template.replace("{{QUESTION}}", q) for q in questions]

        if server_url and model_name:
            # Using vLLM server via HTTP API
            response_texts = []
            for q in query:
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": q
                        }
                    ],
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_tokens": 800
                }
                try:
                    resp = requests.post(
                        server_url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=60  # Optional: set a timeout for the request
                    )
                    resp.raise_for_status()  # Raise an error for bad status codes
                    resp_json = resp.json()
                    generated = resp_json['choices'][0]['message']['content'].strip()
                    response_texts.append(generated)
                except requests.exceptions.RequestException as e:
                    print(f"Request failed for query: {q}\nError: {e}")
                    response_texts.append("")  # Append empty string or handle as needed

        elif "openai/gpt" in args.model:
            assert args.batch_size == 1
            response = [model.complete(query)]
            response_texts = response
        elif args.togetherapi:
            assert args.batch_size == 1
            prompt = "<s>user: " + query[0] + "</s>\nassistant: "
            response = model.safe_completion(prompt, check_prompt=False)["content"]
            response_texts = [response]
        else:
            query, stop = generation_utils(query, args, tokenizer)
            inputs = tokenizer(query, add_special_tokens=False, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    inputs=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    pad_token_id=tokenizer.eos_token_id, 
                    stopping_criteria=stop, 
                    max_new_tokens=800
                )
            out = out[:, inputs["input_ids"].shape[1]:]
            response_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        
        sample['query'] = query
        sample["output"] = response_texts
        sample["model"] = [args.model] * len(questions)
        sample["closedbook_eval"] = [args.closedbook] * len(questions)
        sample["hf_chat_template"] = [args.hf_chat_template] * len(questions)
        sample["bnb4bit"] = [args.bnb4bit] * len(questions)
        outputs += [{k: sample[k][i] for k in sample.keys()} for i in range(len(sample["output"]))]
        
        if len(outputs) >= 1000:
            save_outputs(outputs, base_save, batch_index, args.ddp_rank)
            outputs = []
            batch_index += 1

    # Save any remaining outputs
    if outputs:
        save_outputs(outputs, base_save, batch_index, args.ddp_rank)
        
    return outputs

if __name__ == "__main__":
    print("Current SLURM_PROCID:", os.getenv('SLURM_PROCID'))
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="princeton-nlp/Llemma-7B-32K-MathMix", type=str, help="Generator model")
    parser.add_argument("--output_dir", default="tutoreval/generations", type=str, help="Output directory for generations")
    parser.add_argument("--closedbook", action="store_true", help="Use closed book evaluation")
    parser.add_argument("--hf_chat_template", action="store_true", help="Use the chat template from tokenizer")
    parser.add_argument("--togetherapi", action="store_true", help="Use the TogetherAI API")
    parser.add_argument("--rope_theta", default=-1, type=int, help="Set RoPE theta for context window extension")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size used during generation")
    parser.add_argument("--ddp_worldsize", default=1, type=int, help="Number of parallel instances for data parallelism")
    parser.add_argument("--ddp_rank", default=0, type=int, help="Rank of the data fragment for generation")
    parser.add_argument("--bnb4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--vllm", action="store_true", help="Use vLLM for generation")
    args = parser.parse_args()

    #print(args.ddp_rank)

    # Load data
    data = load_dataset("meta-math/MetaMathQA")['train'].select(range(10))

    if args.closedbook:
        with open("templates/closedbook_generation_template.txt", "r") as f:
            template = f.read()
    else:
        with open("templates/generation_template.txt", "r") as f:
            template = f.read()

    if args.ddp_worldsize > 1:
        assert args.ddp_rank in range(args.ddp_worldsize)
        data = data.select(list(range(args.ddp_rank, len(data), args.ddp_worldsize)))
    data = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)

    if args.vllm:
        # Configure vLLM server parameters
        server_url = "http://localhost:8000/v1/chat/completions"  # Ensure the vLLM server is running at this URL
        model_name = args.model  # The model name to request from the server
        tokenizer = None  # Tokenizer not needed when using the server
        # No model object is instantiated since requests are made to the server
    elif "openai/gpt" in args.model:
        engine = args.model.split("/")[1]
        args.system_prompt = "You are a helpful science teacher..."
        model = OpenAI(model=engine, system_prompt=args.system_prompt)
        tokenizer = None
        args.batch_size = 1
    elif args.togetherapi:
        model = TogetherBaseEngine(args.model)
        tokenizer = None
        args.batch_size = 1
    else:
        config = AutoConfig.from_pretrained(args.model)
        config.max_new_tokens = 800
        config.dtype = torch.bfloat16
        config.do_sample = False
        config.use_cache = True
        if args.rope_theta != -1:
            config.rope_theta = args.rope_theta
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        if args.bnb4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        model.eval()

    base_save = f"{args.output_dir}/{'closedbook' if args.closedbook else 'openbook'}"
    if args.vllm:
        outputs = generate_answers(data, template, model=None, tokenizer=None, server_url=server_url, model_name=model_name)
    else:
        outputs = generate_answers(data, template, model, tokenizer)
