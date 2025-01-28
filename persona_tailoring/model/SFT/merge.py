"""
Merges the SFT LoRA weights with the original model
"""

from model.util import TrainingType
from model import config

import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM
import time
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SFT Merging Script")
    parser.add_argument(
        '--training_type', 
        type=str, 
        required=True, 
        choices=[t.value for t in TrainingType], 
        help="Type of training (e.g., 'chosen', 'rejected')"
    )
    return parser.parse_args()

def main(args):
    start_time = time.time()
    
    final_model_name = f"{config.params['sft_final_output_dir']}_{args.training_type}"
    adapter_name = f"{config.params['sft_adapter_name']}_{args.training_type}"
    model_name = config.params['base_model_name']
    cache_dir = config.params['cache_dir']

    # Check that models exist
    for file in [adapter_name]:
        model_path = Path(file)
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model directory '{file}' does not exist. Please ensure the SFT model is trained or provide the correct arguments.")

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                cache_dir=cache_dir,
                                                device_map="cpu")
    print(f'loaded base model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    model = PeftModel.from_pretrained(model, adapter_name, device_map='cpu')
    print(f'loaded PEFT model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    model = model.merge_and_unload()
    print(f'merged model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    model.save_pretrained(final_model_name)
    print(f'done saving in {time.time() - start_time} seconds', flush=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)