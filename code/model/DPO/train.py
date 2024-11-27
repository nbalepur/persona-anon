import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import config
from data_loader import load_dpo_data
from trl import DPOTrainer, DPOConfig
import argparse
import wandb


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
Response:""" for idx in range(len(ex['prompt']))]}
        else:
            return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Response:""" for idx in range(len(ex['prompt']))]}
    
    run_name = f'DPO_{config.params["model_nickname"]}_{use_persona}_{response_type}'
    os.environ["WANDB_PROJECT"] = config.params['wandb_name'] + f"-{config.params['dataset_name']}"

    sft_final_model_name = f"{config.params['sft_final_output_dir']}_{use_persona}_{response_type}"
    
    tokenizer_name = config.params['sft_tokenizer_name']
    
    cache_dir = config.params['cache_dir']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)

    device_map = config.device_map
    model = AutoModelForCausalLM.from_pretrained(
        sft_final_model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        cache_dir=cache_dir,
        device_map=device_map,
    )

    ds_dpo_train, ds_dpo_eval = load_dpo_data(use_persona, response_type)
    ds_dpo_train_ = ds_dpo_train.map(create_exemplar, batched=True)
    ds_dpo_eval_ = ds_dpo_eval.map(create_exemplar, batched=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        num_train_epochs=10, 
        overwrite_output_dir=True,
        output_dir=config.params['dpo_output_dir'],
        metric_for_best_model='eval_loss',
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        per_device_train_batch_size=1,
        learning_rate=5e-06,
        load_best_model_at_end=True,
        report_to='wandb',
        run_name=run_name,
        max_length=512,
        max_prompt_length=128,
        beta=0.1,
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=ds_dpo_train_,
        eval_dataset=ds_dpo_eval_,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    dpo_trainer.train()
    dpo_adapter_name = f"{config.params['dpo_adapter_name']}_{use_persona}_{response_type}"
    dpo_trainer.save_model(dpo_adapter_name)

if __name__ == '__main__':
    main()