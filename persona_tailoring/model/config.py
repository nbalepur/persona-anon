import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Generation Parameters
STATIC_PARAMS = {
    'model_nickname': 'llama_1b',
    'base_model_name': 'meta-llama/Llama-3.2-1B',
    'use_wandb': True,
    'dataset_name': 'BeaverTails',
    'load_in_8bit': False,
    'load_in_4bit': False,
    'cache_dir': '',
    'model_save_dir': '',
    'results_save_dir': 'results/',
    'device_map': 'auto'
}

# API Keys (from .env)
SENSITIVE_KEYS = {
    'wandb_key': os.getenv('WANDB_KEY', ''),
    'wandb_name': 'persona-tailoring',
    'hf_read_token': os.getenv('HF_READ_TOKEN', ''),
}

# Dynamic parameters
def generate_dynamic_params(params):
    model_save_dir = Path(params['model_save_dir'])
    results_save_dir = Path(params['results_save_dir'])
    dataset_name = params['dataset_name']
    model_nickname = params['model_nickname']

    return {
        'sft_output_dir': model_save_dir / f'{dataset_name}/sft_{model_nickname}',
        'sft_final_output_dir': model_save_dir / f'{dataset_name}/sft_{model_nickname}_final',
        'dpo_output_dir': model_save_dir / f'{dataset_name}/dpo_{model_nickname}',
        'dpo_final_output_dir': model_save_dir / f'{dataset_name}/dpo_{model_nickname}_final',
        'sft_adapter_name': model_save_dir / f'{dataset_name}/sft_{model_nickname}_adapter',
        'sft_tokenizer_name': model_save_dir / f'{dataset_name}/sft_{model_nickname}_tokenizer',
        'dpo_adapter_name': model_save_dir / f'{dataset_name}/dpo_{model_nickname}_adapter',
        'fewshot_results_dir': results_save_dir / f'{dataset_name}/{model_nickname}/fewshot',
        'sft_results_dir': results_save_dir / f'{dataset_name}/{model_nickname}/sft',
        'dpo_results_dir': results_save_dir / f'{dataset_name}/{model_nickname}/dpo',
    }

# Merge parameters
params = {**STATIC_PARAMS, **SENSITIVE_KEYS, **generate_dynamic_params(STATIC_PARAMS)}