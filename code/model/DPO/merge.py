import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import time
import argparse

def main():

    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument('--use_persona', type=str, default="False", help='Use persona in prompts (True/False)')
    parser.add_argument('--response_type', type=str, default='chosen', help="Response type, e.g., 'chosen'")
    args = parser.parse_args()

    use_persona = args.use_persona == "True"
    response_type = args.response_type

    start_time = time.time()

    cache_dir = config.params['cache_dir']

    sft_model_name = f"{config.params['sft_final_output_dir']}_{use_persona}_{response_type}"
    dpo_adapter_name = f"{config.params['dpo_adapter_name']}_{use_persona}_{response_type}"
    tokenizer_name = config.params['sft_tokenizer_name']
    final_model_name = f"{config.params['dpo_final_output_dir']}_{use_persona}_{response_type}"

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
    main()