from model.data_loader import DataLoader
from model.prompt_loader import fetch_training_template, fetch_testing_template
from model.util import StoppingCriteriaSub, Persona, TrainingType, InferenceType, ModelType
from model import config

import json
import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, logging
from huggingface_hub import login
import argparse

logging.set_verbosity_error()
login(token=config.params['hf_read_token'])

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot Inference Script")
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


def main(args):
    persona = Persona(
        training_type=TrainingType(args.training_type),
        inference_type=InferenceType(args.inference_type)
    )
    dataset_split = config.params['dataset_name']

    # Initialize DataLoader
    dl = DataLoader(dataset_split)

    # Load tokenizer and model
    cache_dir = config.params['cache_dir']
    tokenizer_name = config.params['base_model_name']
    fewshot_model_name = config.params['base_model_name']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    fewshot_model = AutoModelForCausalLM.from_pretrained(
        fewshot_model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        cache_dir=cache_dir,
        device_map="auto"
    )

    # Load train and test data
    train = dl.load_few_shot_data(persona)
    train_prompt_template = fetch_training_template(training_type=persona.training_type, model_type=ModelType.fewshot)
    train = train.map(train_prompt_template, batched=True)
    
    test = dl.load_test_data(persona)
    test_prompt_template = fetch_testing_template(training_type=persona.training_type, inference_type=persona.inference_type, dataset_split=dataset_split)
    test = test.map(test_prompt_template, batched=True)

    # Generate inference prompts
    base_prompt = '\n\n'.join(train['prompt'])
    inf_prompts = [base_prompt + '\n\n' + p for p in test['prompt']]

    # Prepare output directory
    fewshot_results_dir = config.params['fewshot_results_dir']
    fewshot_results_path = Path(f"{fewshot_results_dir}_{persona}.jsonl")
    fewshot_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    outputs = []
    stop_token = '\nPrompt:'
    for idx in tqdm.tqdm(range(len(inf_prompts))):
        prompt = inf_prompts[idx]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

        stopping_criteria = StoppingCriteriaList([
            StoppingCriteriaSub(
                tokenizer(stop_token).input_ids[2:], 
                prompt_len=input_ids.shape[1]
            )
        ])

        # Generate output
        out = fewshot_model.generate(
            input_ids, max_new_tokens=512, do_sample=False, stopping_criteria=stopping_criteria
        ).to('cpu').detach()
        out = out[:, input_ids.shape[1]:]
        out = tokenizer.batch_decode(out)

        # Save intermediate results
        outputs.append({'response': out[0], 'prompt': prompt})
        with fewshot_results_path.open('w') as fout:
            json.dump(outputs, fout)

if __name__ == '__main__':
    args = parse_args()
    main(args)