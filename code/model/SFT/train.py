import sys
import argparse
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import datasets
from huggingface_hub.hf_api import HfFolder
from trl import SFTTrainer, SFTConfig
import config
from peft import LoraConfig
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from data_loader import load_sft_data
import wandb

ALL_USE_PERSONA = [True, False]
ALL_RESPONSE_TYPE = ['chosen']

def main():

    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument('--use_persona', type=str, default="False", help='Use persona in prompts (True/False)')
    parser.add_argument('--response_type', type=str, default='chosen', help="Response type, e.g., 'chosen'")
    args = parser.parse_args()

    use_persona = args.use_persona == "True"
    response_type = args.response_type
    
    def create_exemplar(ex):
        if use_persona:
            return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
Response: {ex['response'][idx]}<|end_of_text|>""" for idx in range(len(ex['prompt']))]}
        else:
            return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Response: {ex['response'][idx]}<|end_of_text|>""" for idx in range(len(ex['prompt']))]}
    
    run_name = f'SFT_{config.params["model_nickname"]}_{use_persona}_{response_type}'
    os.environ["WANDB_PROJECT"] = config.params['wandb_name'] + f"-{config.params['dataset_name']}"

    hf_token = config.params['hf_read_token']
    model_name = config.params['base_model_name']
    cache_dir = config.params['cache_dir']
    
    adapter_name = f"{config.params['sft_adapter_name']}_{use_persona}_{response_type}"
    model_save_dir = f"{config.params['sft_output_dir']}_{use_persona}_{response_type}"
    
    tokenizer_name = config.params['sft_tokenizer_name']
    HfFolder.save_token(hf_token)

    ds_sft_train, ds_sft_eval = load_sft_data(use_persona, response_type)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        device_map="auto",
        cache_dir=cache_dir
    )
    
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds_sft_train, ds_sft_eval = ds_sft_train.map(create_exemplar, batched=True), ds_sft_eval.map(create_exemplar, batched=True)

    print(ds_sft_train['prompt'][0])
    exit(0)
    
    sft_config = SFTConfig(
        dataset_text_field="prompt",
        max_seq_length=512,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        output_dir=model_save_dir,
        num_train_epochs=10, 
        overwrite_output_dir=True,
        logging_strategy='epoch',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='wandb',
        learning_rate=2e-05,
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=ds_sft_train,
        eval_dataset=ds_sft_eval,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(adapter_name)
    tokenizer.save_pretrained(tokenizer_name)
    

if __name__ == '__main__':
    main()