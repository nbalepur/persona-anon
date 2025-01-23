from model.data_loader import DataLoader
from model.prompt_loader import fetch_testing_template
from model.util import StoppingCriteriaSub, TrainingType, InferenceType, Persona
from model import config

import argparse
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from pathlib import Path
import json

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SFT Inference Script")
    parser.add_argument(
        '--training_type',
        type=str,
        required=True,
        choices=[t.value for t in TrainingType],
        help="Type of training (e.g., 'chosen', 'rejected')"
    )
    parser.add_argument(
        '--inference_type',
        type=str,
        required=True,
        choices=[t.value for t in InferenceType],
        help="Type of inference (e.g., 'chosen', 'rejected')"
    )
    return parser.parse_args()

def setup_model_and_tokenizer(persona: Persona, cache_dir: str):
    """Load the model and tokenizer for the specified persona."""
    sft_model_name = f"{config.params['sft_final_output_dir']}_{persona.training_type.value}"
    tokenizer_name = config.params['sft_tokenizer_name']

    # Check that models exist
    for file in [tokenizer_name, sft_model_name]:
        model_path = Path(file)
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model directory '{file}' does not exist. Please ensure the SFT model is trained or provide the correct arguments.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        cache_dir=cache_dir,
        device_map="auto"
    )
    return model, tokenizer

def create_output_directory():
    """Ensure the results directory exists."""
    results_dir = Path(config.params['sft_results_dir']).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def generate_responses(tokenizer, model, inf_prompts, stop_token, output_file):
    """Generate responses for the given prompts and save them."""
    outputs = []
    for idx in tqdm.tqdm(range(len(inf_prompts))):
        prompt = inf_prompts[idx]
        encoding = tokenizer(prompt, return_tensors='pt')
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')

        stopping_criteria = StoppingCriteriaList([
            StoppingCriteriaSub(
                tokenizer(stop_token).input_ids[2:], 
                prompt_len=input_ids.shape[1]
            )
        ])

        # Generate output
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=10,
            max_new_tokens=512,
            do_sample=False,
            stopping_criteria=stopping_criteria
        ).to('cpu').detach()
        out = out[:, input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(out)
        
        # Save result
        outputs.append({'response': decoded[0], 'prompt': prompt})
        with output_file.open('w') as fout:
            json.dump(outputs, fout)

def main(args):
    """Main script logic."""
    persona = Persona(
        training_type=TrainingType(args.training_type),
        inference_type=InferenceType(args.inference_type)
    )
    cache_dir = config.params['cache_dir']

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(persona, cache_dir)

    # Load test data and apply template
    dl = DataLoader(config.params['dataset_name'])
    test_prompt_template = fetch_testing_template(training_type=args.training_type, dataset_split=config.params['dataset_name'])
    ds_test = dl.load_test_data(persona).map(test_prompt_template, batched=True)
    inf_prompts = ds_test['prompt']

    # Ensure output directory exists
    create_output_directory()
    
    # Generate text
    output_file = Path(f"{config.params['sft_results_dir']}_{persona}.jsonl")
    stop_token = '\nPrompt:'
    generate_responses(tokenizer, model, inf_prompts, stop_token, output_file)

if __name__ == '__main__':
    args = parse_args()
    main(args)