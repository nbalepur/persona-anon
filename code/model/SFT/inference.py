import sys
import os
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import datasets
import tqdm
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import config
from data_loader import load_test_data
from peft import PeftModel
import torch
import json

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens = [], prompt_len = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist)+i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen

def main():

    parser = argparse.ArgumentParser(description="SFT Inference Script")
    parser.add_argument('--use_persona', type=str, default="False", help='Use persona in prompts (True/False)')
    parser.add_argument('--response_type', type=str, default='chosen', help="Response type, e.g., 'chosen'")
    args = parser.parse_args()

    use_persona = args.use_persona == "True"
    response_type = args.response_type

    cache_dir = config.params['cache_dir']

    sft_model_name = f"{config.params['sft_final_output_dir']}_{use_persona}_{response_type}"
    tokenizer_name = config.params['sft_tokenizer_name']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_name,
                                                    load_in_8bit=config.params['load_in_8bit'],
                                                    load_in_4bit=config.params['load_in_4bit'],
                                                    cache_dir=cache_dir,
                                                    device_map="auto")

    sft_model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    #sft_model.config.pad_token_id = tokenizer.eos_token_id
    
    def create_exemplar_test(ex):
        response_prefix = [f"Response: {ex['prompt'][idx].title()} sounds like" if config.params['dataset_name'] == 'Mnemonic' else "Response:" for idx in range(len(ex['prompt']))]
        if use_persona:
            return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
{response_prefix[idx]}""" for idx in range(len(ex['prompt']))]}
        else:
            return {'prompt': [f"""Prompt: {ex['prompt'][idx]} {ex['persona'][idx]}
{response_prefix[idx]}""" for idx in range(len(ex['prompt']))]}
            

    if config.params['dataset_name'] == 'Mnemonic':    
        INFERENCE_TYPE = ['system', 'none'] if use_persona else ['none', 'system', 'retr-rejected', 'retr']
    else:
        INFERENCE_TYPE = ['system', 'none'] if use_persona else ['none', 'system', 'retr-rejected', 'retr', 'oracle', 'oracle-rejected']

    INFERENCE_TYPE = ['system']
    
    for inference_type in INFERENCE_TYPE:
    
        ds_test = load_test_data(use_persona, response_type, inference_type).map(create_exemplar_test, batched=True)
        inf_prompts = ds_test['prompt']
    
        path = '/'.join(config.params['sft_results_dir'].split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)
    
        f = f"{config.params['sft_results_dir']}_{use_persona}_{inference_type}_{response_type}.jsonl"
    
        stop_token = '\nPrompt:'
        outputs = []
        for idx in tqdm.tqdm(range(len(outputs), len(inf_prompts))):
    
            prompt = inf_prompts[idx]
            encoding = tokenizer(prompt, return_tensors='pt')
            input_ids = encoding['input_ids'].to('cuda')
            attention_mask = encoding['attention_mask'].to('cuda')
            
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:], prompt_len=input_ids.shape[1])])
            
            out = sft_model.generate(input_ids=input_ids, attention_mask=attention_mask, min_new_tokens=10, max_new_tokens=512, do_sample=False, stopping_criteria=stopping_criteria).to('cpu').detach()
            out = out[:, input_ids.shape[1]:]
            out = tokenizer.batch_decode(out)
            
            outputs.append({'response': out[0], 'prompt': prompt})
            with open(f, 'w') as fout:
                json.dump(outputs, fout)

if __name__ == '__main__':
    main()