"""
Shared utilities for loading and processing MetaMathQA dataset.
Used across all training scripts for consistency.
"""

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os

def load_metamathqa_dataset(split="train[:1000]", cache_dir="./metamathqa_tokenized_data", rank=0):
    """
    Load and tokenize MetaMathQA dataset for Qwen 2.5 instruction tuning.
    
    Args:
        split: Dataset split (e.g., "train[:1000]" for testing)
        cache_dir: Directory to cache tokenized dataset
        rank: Process rank (for distributed training)
    
    Returns:
        tokenized_dataset: Tokenized dataset ready for training
        tokenizer: Qwen tokenizer
    """
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if cached dataset exists
    if os.path.exists(cache_dir) and rank == 0:
        print(f"Loading cached tokenized dataset from {cache_dir}...")
        try:
            tokenized_dataset = load_from_disk(cache_dir)
            print(f"✅ Loaded {len(tokenized_dataset)} examples from cache")
            return tokenized_dataset, tokenizer
        except:
            print("Cache load failed, regenerating...")
    
    # Load and process dataset (only on rank 0)
    if rank == 0:
        print("Loading MetaMathQA dataset...")
        print("Dataset: https://huggingface.co/datasets/meta-math/MetaMathQA")
        print("Size: ~395K math QA examples")
        
        dataset = load_dataset("meta-math/MetaMathQA", split=split)
        
        # Format for Qwen instruction tuning
        def format_qwen_instruction(question, answer):
            """Format MetaMathQA data for Qwen 2.5 instruction tuning."""
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that solves mathematical problems step by step."
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        def tokenize_function(examples):
            """Tokenize MetaMathQA examples for training."""
            texts = []
            for question, answer in zip(examples["query"], examples["response"]):
                formatted = format_qwen_instruction(question, answer)
                texts.append(formatted)
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=2048,  # Longer for math problems
                padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["type", "query", "original_question", "response"]
        )
        
        # Save to cache
        print(f"Saving tokenized dataset to {cache_dir}...")
        tokenized_dataset.save_to_disk(cache_dir)
        print(f"✅ Dataset processed: {len(tokenized_dataset)} examples")
    
    # For non-rank-0 processes, wait and load from cache
    if rank != 0:
        import torch.distributed as dist
        dist.barrier()
        tokenized_dataset = load_from_disk(cache_dir)
    
    return tokenized_dataset, tokenizer

# Test loading
if __name__ == "__main__":
    dataset, tokenizer = load_metamathqa_dataset()
    print(f"Dataset keys: {dataset[0].keys()}")
    print(f"Sequence length: {len(dataset[0]['input_ids'])}")

