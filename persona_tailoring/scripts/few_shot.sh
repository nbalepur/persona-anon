#!/bin/bash
# source ... -> activate your conda environment
conda activate persona

training_type="chosen"
inference_type="gold-chosen"

python3 -m model.Few_Shot.inference \
--training_type="$training_type" \
--inference_type="$inference_type"