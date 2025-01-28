#!/bin/bash

# source ... -> activate your conda environment
conda activate persona

dataset_name="nbalepur/persona-inference" # original dataset
inference_split="BeaverTails" # inference split used to create personas

res_dir="results/" # results folder directory
run_name="default" # run name of personas
model="gpt-4o-mini" # model that generated personas

new_dataset_name="my_name/BeaverTails_with_Persona" # new name of the dataset

# API tokens
if [ -f ../.env ]; then
    export $(cat ../.env | xargs)
fi
hf_write_token="${HF_WRITE_TOKEN:-}"

# --push_to_hub is a `True` flag denoting whether you want to upload the dataset to the huggingface hub
python3 model/build_augmented_dataset.py \
--run_name="$run_name" \
--model="$model" \
--res_dir="$res_dir" \
--dataset_name="$dataset_name" \
--inference_split="$inference_split" \
--new_dataset_name="$new_dataset_name" \
--hf_write_token="$hf_write_token" \
--push_to_hub