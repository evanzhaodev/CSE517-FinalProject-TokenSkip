## Model Weights

Download corresponding model weights and modify the checkpoint path in `eval.sh`.

| LoRA Adapter                         | Link                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| TokenSkip-Qwen2.5-3B-Instruct-GSM8K  | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K) |
| TokenSkip-Qwen2.5-7B-Instruct-GSM8K  | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K) |
| TokenSkip-Qwen2.5-14B-Instruct-GSM8K | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-14B-Instruct-GSM8K) |

## Installation

```
conda create -n tokenskip python=3.12
conda activate tokenskip
cd TokenSkip
pip install -r requirements.txt
```

## Token Pruning

**1.Obtain the original CoT outputs of the training data, using the target LLM**

run `evaluation` with correct model path and tokenizer path.

```
python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/" \
    --model-path "/your_model_path/Qwen2.5-7B-Instruct" --tokenizer-path ${MODEL_PATH} \
    --model-size "7b" --model-type "qwen" --data-type "train"  \
    --max_num_examples 100000000000000 --max_new_tokens 512 \
    --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k"
```

> The original CoT outputs of the target LLM will be stored in `outputs/.../Original`.

**2.Prune original CoTs using LLMLingua**

Download the [model weights](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) for [LLMLingua-2](https://github.com/microsoft/LLMLingua) and modify the checkpoint path in `LLMLingua.py`.

Run `LLMLingua` to obtain compressed CoTs with various compression ratios.

```
python ./LLMLingua.py
```

> The compressed CoTs will be stored in `outputs/.../Compression`.

**3.Convert training data to LLaMA-Factory format**

Run `get_llamafactory_input` to convert the training data into the format of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```
python ./get_llamafactory_input.py
```

> The converted data will be stored in `outputs/mydataset.json`.
>
> For reference, we provide our processed training data in `datasets/gsm8k/llamafactory_inputs/`.

## Training

TokenSkip follows the general LoRA SFT pipeline of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Here's how to set it up:

1. Git clone [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and install the required environments.
2. Place the training data under `LLaMA-Factory/data/` and register it in `data/dataset_info.json`.
3. To fine-tune the target LLM with LoRA, run the following command:

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen.yaml
```

> We provide our training configs in `configs/examples/train_lora` for your reference.

## Inference

Run `evaluation` with the correct model path, tokenizer path, and adapter path.

```
python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/" \
    --model-path "/your_model_path/Qwen2.5-7B-Instruct" --tokenizer-path ${MODEL_PATH} \
    --model-size "7b" --model-type "qwen" --data-type "test"  \
    --max_num_examples 100000000000000 --max_new_tokens 512 \
    --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k" \
    --adapter-path "/your_model_path/TokenSkip-Qwen2.5-7B-Instruct-GSM8K" \
    --compression_ratio 0.5 --use_adapter
```

## Custom Extension: Token Recovery Evaluation

We extended the codebase with an interactive script (recovery_eval.py) to test whether the LLMs are capable of restoring the CoT process from compressed outputs.

Run `recovery_eval` with the correct model path.

```
python recovery_eval.py --model-path "Qwen/Qwen2.5-7B-Instruct" --max_new_tokens 512
```

Usage: The script prompts for the original math question, compression rate, and the pruned text sequence. It then instructs the model to act as a decompression algorithm by restoring the missing semantic connectors and formatting, logging the results to recovery_logs.jsonl for exact-match analysis.
