"""
Merges the DPO LoRA weights with the original model
"""

from model.util import TrainingType
from model import config

from peft import PeftModel
from transformers import AutoModelForCausalLM
import time
import argparse
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DPO Merging Script")
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
    cache_dir = config.params['cache_dir']

    sft_model_name = f"{config.params['sft_final_output_dir']}_{args.training_type}"
    dpo_adapter_name = f"{config.params['dpo_adapter_name']}_{args.training_type}"
    final_model_name = f"{config.params['dpo_final_output_dir']}_{args.training_type}"

    # Check that models exist
    for file in [sft_model_name, dpo_adapter_name]:
        model_path = Path(file)
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model directory '{file}' does not exist. Please ensure the SFT/DPO models are trained or provide the correct arguments.")

    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_name,
                                                cache_dir=cache_dir,
                                                device_map="auto")
    print(f'loaded base model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    dpo_model = PeftModel.from_pretrained(sft_model, dpo_adapter_name)
    print(f'loaded PEFT model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    dpo_model = dpo_model.merge_and_unload()
    print(f'merged model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    dpo_model.save_pretrained(final_model_name)
    print(f'done saving in {time.time() - start_time} seconds', flush=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)