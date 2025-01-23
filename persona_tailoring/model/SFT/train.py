from model.data_loader import DataLoader
from model.prompt_loader import fetch_training_template
from model.util import TrainingType
from model import config

import argparse
import os
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        '--training_type', 
        type=str, 
        required=True, 
        choices=[t.value for t in TrainingType], 
        help="Type of training (e.g., 'chosen', 'rejected')"
    )
    return parser.parse_args()

def main(args):
    # Define run name and directories
    run_name = f"SFT_{config.params['model_nickname']}_{args.training_type}"
    os.environ["WANDB_PROJECT"] = f"{config.params['wandb_name']}-{config.params['dataset_name']}"

    model_name = config.params['base_model_name']
    cache_dir = config.params['cache_dir']
    adapter_name = f"{config.params['sft_adapter_name']}_{args.training_type}"
    model_save_dir = f"{config.params['sft_output_dir']}_{args.training_type}"
    tokenizer_name = config.params['sft_tokenizer_name']

    # Initialize DataLoader
    dl = DataLoader(config.params['dataset_name'])

    # Load train and evaluation datasets
    ds_sft_train, ds_sft_eval = dl.load_sft_data(TrainingType(args.training_type))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        device_map="auto",
        cache_dir=cache_dir
    )

    # Add special tokens
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply persona-based prompts
    train_prompt_template = fetch_training_template(training_type=TrainingType(args.training_type), add_eot=True, add_response=True)
    ds_sft_train = ds_sft_train.map(train_prompt_template, batched=True)
    ds_sft_eval = ds_sft_eval.map(train_prompt_template, batched=True)

    # Configure SFT Training
    sft_config = SFTConfig(
        dataset_text_field="prompt",
        max_seq_length=512,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        output_dir=model_save_dir,
        num_train_epochs=10,
        overwrite_output_dir=True,
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='wandb',
        learning_rate=2e-5,
        run_name=run_name,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=ds_sft_train,
        eval_dataset=ds_sft_eval,
        peft_config=peft_config,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model(adapter_name)
    tokenizer.save_pretrained(tokenizer_name)

if __name__ == '__main__':
    args = parse_args()
    main(args)