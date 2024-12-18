#!/bin/bash
# source ... -> activate your conda environment
conda activate persona

model_type_base="dpo"
training_type_base="chosen"
inference_type_base="gold-chosen"

model_type_test="sft"
training_type_test="chosen"
inference_type_test="gold-chosen"

python3 -m evaluation.run_judge \
--model_type_base="$model_type_base" \
--training_type_base="$training_type_base" \
--inference_type_base="$inference_type_base" \
--model_type_test="$model_type_test" \
--training_type_test="$training_type_test" \
--inference_type_test="$inference_type_test"

python3 -m evaluation.win_rate \
--model_type_base="$model_type_base" \
--training_type_base="$training_type_base" \
--inference_type_base="$inference_type_base" \
--model_type_test="$model_type_test" \
--training_type_test="$training_type_test" \
--inference_type_test="$inference_type_test"