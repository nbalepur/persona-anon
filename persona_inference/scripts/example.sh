#!/bin/bash
# source ... -> activate your conda environment
conda activate persona

# dataset details
inference_split="Mnemonic" # split of the dataset to use
dataset_name="nbalepur/persona-inference" # huggingface preference dataset, requires columns with `prompt`, `chosen`, `rejected`
partition="full" # partition of the dataset to run

# model details
model_type="open_ai" # type of the model (see ModelType in enums.py)
model_name="gpt-4o-mini" # endpoint for the model

# how to identify this run
run_name="default"

# experiments (see PromptType in enums.py)
experiments=("persona_inference")
experiments_str=$(IFS=" "; echo "${experiments[*]}")

# model generation parameters
num_shots=0
temperature=0.0
min_tokens=5
max_tokens=200
stop_token="\nPrompt:"
device_map="auto" # where to load tensors ('cpu', 'cuda', 'auto')

# API tokens
if [ -f ../.env ]; then
    export $(cat ../.env | xargs)
fi
hf_token="${HF_READ_TOKEN:-}"
open_ai_token="${OPEN_AI_TOKEN:-}"
cohere_token="${COHERE_TOKEN:-}"
anthropic_token="${ANTHROPIC_TOKEN:-}"

# directory setup
res_dir="results/" # Results folder directory
prompt_dir="prompts/" # Prompt folder directory
cache_dir="" # Cache directory to save any models

# there are also --True flags for `load_in_4bit` and `load_in_8bit`
python3 model/run_model.py \
--run_name="$run_name" \
--model_nickname="$model_name" \
--model_name="$model_name" \
--model_type="$model_type" \
--dataset_name="$dataset_name" \
--inference_split="$inference_split" \
--partition="$partition" \
--hf_token="$hf_token" \
--llm_proxy_token="$llm_proxy_token" \
--open_ai_token="$open_ai_token" \
--cohere_token="$cohere_token" \
--anthropic_token="$anthropic_token" \
--device_map="$device_map" \
--num_shots="$num_shots" \
--temperature="$temperature" \
--min_tokens="$min_tokens" \
--max_tokens="$max_tokens" \
--stop_token="$stop_token" \
--prompt_types="$experiments_str" \
--res_dir="$res_dir" \
--prompt_dir="$prompt_dir" \
--cache_dir="$cache_dir"