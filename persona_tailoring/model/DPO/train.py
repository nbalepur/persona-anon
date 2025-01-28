"""
Trains the DPO model
"""

from model.data_loader import DataLoader
from model.prompt_loader import fetch_training_template
from model.util import TrainingType, ModelType
from model import config

import argparse
from pathlib import Path
import os
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DPO Training Script")
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
    run_name = f"DPO_{config.params['model_nickname']}_{args.training_type}"
    os.environ["WANDB_PROJECT"] = f"{config.params['wandb_name']}-{config.params['dataset_name']}"

    # Define model and directories
    sft_final_model_name = f"{config.params['sft_final_output_dir']}_{args.training_type}"
    tokenizer_name = config.params['sft_tokenizer_name']
    cache_dir = config.params['cache_dir']

    # Check that SFT model exists
    model_path = Path(sft_final_model_name)
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(f"Model directory '{sft_final_model_name}' does not exist. Please ensure the SFT model is trained or provide the correct arguments.")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        sft_final_model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        cache_dir=cache_dir,
        device_map="auto",
    )

    # Initialize DataLoader and load datasets
    dl = DataLoader(config.params['dataset_name'])
    ds_dpo_train, ds_dpo_eval = dl.load_dpo_data(TrainingType(args.training_type))

    prompt_training_template = fetch_training_template(training_type=TrainingType(args.training_type), model_type=ModelType.dpo)
    ds_dpo_train = ds_dpo_train.map(prompt_training_template, batched=True).select_columns(['prompt', 'chosen', 'rejected'])
    ds_dpo_eval = ds_dpo_eval.map(prompt_training_template, batched=True).select_columns(['prompt', 'chosen', 'rejected'])

    # Configure DPO Training
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
        output_dir=f"{config.params['dpo_output_dir']}_{args.training_type}",
        metric_for_best_model="eval_loss",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=1,
        learning_rate=5e-6,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        max_length=512,
        max_prompt_length=128,
        beta=0.1,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=ds_dpo_train,
        eval_dataset=ds_dpo_eval,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train and save the model
    dpo_trainer.train()
    dpo_adapter_name = f"{config.params['dpo_adapter_name']}_{args.training_type}"
    dpo_trainer.save_model(dpo_adapter_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)
