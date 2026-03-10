import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    print(f"Loading model and tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def recover_tokens(model, tokenizer, question, pruned_text, compression_rate, max_new_tokens, temperature):
    # Convert string to float for the percentage calculation
    try:
        rate_float = float(compression_rate)
        percentage = rate_float * 100
    except ValueError:
        percentage = 50.0 # Fallback if parsing fails

    prompt = (
        "Could you please recover the following compressed Chain-of-Thought output of a mathematical question to its original full content?\n"
        f"Note: This sequence was compressed with a retain ratio of {compression_rate}. "
        f"This means roughly {percentage:.0f}% of the original text was kept. "
        "If this percentage is high (e.g., 80%), very few words are missing, so you should only insert a minimal amount of semantic connectors. "
        "If this percentage is lower (e.g., 50%), more grammatical words and formatting need to be restored. "
        "Crucially: Your task is ONLY to restore missing grammatical glue. Do not hallucinate new reasoning steps, alter the math, or over-explain. Do not output anything other than your recovered text. \n\n"
        "The question is:\n"
        f"{question}\n\n"
        "The compressed Chain-of-Thought:\n"
        f"{pruned_text}\n\n"
        "Original Full Chain-of-Thought:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = False if temperature == 0.0 else True

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    recovered_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return recovered_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Hugging Face repo ID or local path")
    parser.add_argument("--log-file", type=str, default="recovery_logs.jsonl", help="Path to save the interaction logs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    print("\nModel loaded successfully. Type 'quit' or 'exit' to stop the script.\n", flush=True)

    while True:
        try:
            question_input = input("Enter the original math question (or 'quit' to exit): ").strip()
            if question_input.lower() in ['quit', 'exit']:
                break

            compression_rate = input("Enter the compression rate (e.g., 0.5, 0.7): ").strip()
            if compression_rate.lower() in ['quit', 'exit']:
                break

            pruned_input = input("Enter the compressed/pruned CoT sequence: ").strip()
            if pruned_input.lower() in ['quit', 'exit']:
                break
            
            original_input = input("Enter the target original CoT sequence (for comparison): ").strip()
            if original_input.lower() in ['quit', 'exit']:
                break
            
            if not pruned_input or not original_input or not question_input or not compression_rate:
                print("Inputs cannot be empty. Please try again.\n")
                continue

            print("Recovering...", flush=True)
            recovered_output = recover_tokens(model, tokenizer, question_input, pruned_input, compression_rate, args.max_new_tokens, args.temperature)
            
            print("\n--- Results ---")
            print(f"Recovered: {recovered_output}")
            print(f"Original:  {original_input}")
            print("---------------\n")
            
            log_entry = {
                "question": question_input,
                "compression_rate": compression_rate,
                "pruned_input": pruned_input,
                "target_original": original_input,
                "model_recovered": recovered_output
            }
            
            with open(args.log_file, "a+", encoding="utf-8") as fout:
                fout.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except KeyboardInterrupt:
            print("\nExiting script.")
            break
