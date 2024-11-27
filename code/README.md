# Persona Generation

This repository contains the code and data for the paper: **When One User’s Trash Another is User’s Treasure: Defeasible Reasoning Enhances Personalization in Large Language Models**

## Models and Dataset

Our datasets are in the `/data/` folder and can be loaded via huggingface:

```python
ds = datasets.load_dataset('/data/dataset')
```

## Code

We provide the code for training persona tailoring models

### Setup

This project was written in Python 3.10.0 and all packages were installed with Anaconda version 23.5.0 and pip version 24.0. All necessary packages can be installed with:

```python
pip install -r requirements.txt
```

### Training Models

We have released the fine-tuning and DPO datasets used in persona tailoring, so you can reproduce our reuslts by running the following code for model training:

1. Navigate to `/model/`
2. Set the specified parameters in `config.py` (described in the file)
2. Run `python SFT/create_initial_model.py`. This creates the initial LLaMA model with an extra token for padding. Requires higher CPU memory (1024 GB for LLaMA-2 70B)
3. Run `python SFT/train.py`. This trains the initial SMART model with supervised fine-tuning and LoRA. Requires higher GPU memory (192 GB for LLaMA-2 70B)
4. Run `python SFT/merge.py`. Merges the LoRA fine-tuned model into the original model. Requires higher CPU memory (1024 GB for LLaMA-2 70B)
5. Run `python DPO/train.py`. Further tunes the initial SMART model with DPO. Requires higher GPU memory (192 GB for LLaMA-2 70B)
6. Run `python DPO/merge.py`. Merges the LoRA DPO model into the original model. Requires higher CPU memory (1024 GB for LLaMA-2 70B)

To run inference with our models you can run:

1. `/model/Few-Shot/inference.py`: Inference with the few-shot models
2. `/model/SFT/inference.py`: Inference with the supervised fine-tuning models
3. `/model/DPO/inference.py`: Inference with the direct preference optimization models


### Evaluation

We provide the script to evaluate using the LLM judge which can be run with:

```python
python evaluate/llm_judge_all.py
```