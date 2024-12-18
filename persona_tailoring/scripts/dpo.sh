#!/bin/bash
# source ... -> activate your conda environment
conda activate persona

training_type="chosen"
inference_type="gold-chosen"

python3 -m model.DPO.train \
--training_type="$training_type"

python3 -m model.DPO.merge \
--training_type="$training_type"

python3 -m model.DPO.inference \
--training_type="$training_type" \
--inference_type="$inference_type"