import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import datasets
from huggingface_hub.hf_api import HfFolder
from trl import SFTTrainer
import config
from peft import LoraConfig
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from data_loader import load_few_shot_data
import wandb
import datasets
import tqdm
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import config
import pickle
from peft import PeftModel
import torch
import json
from huggingface_hub import login
from transformers import logging

logging.set_verbosity_error()

login(token=config.params['hf_read_token'])

# ALL_USE_PERSONA = [True]
# ALL_RESPONSE_TYPE = ['chosen', 'rejected', 'all']

cache_dir = config.params['cache_dir']
tokenizer_name = config.params['base_model_name']
fewshot_model_name = config.params['base_model_name']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
fewshot_model = AutoModelForCausalLM.from_pretrained(fewshot_model_name,
                                                load_in_8bit=config.params['load_in_8bit'],
                                                load_in_4bit=config.params['load_in_4bit'],
                                                cache_dir=cache_dir,
                                                device_map="auto")

for use_persona in [True, False]:
    #ALL_RESPONSE_TYPE = ['chosen', 'rejected', 'all'] if use_persona else ['chosen']
    ALL_RESPONSE_TYPE = ['chosen']
    for response_type in ALL_RESPONSE_TYPE:
    
        if config.params['dataset_name'] == 'Mnemonic':    
            INFERENCE_TYPE = ['none', 'system'] if use_persona else ['none', 'system', 'retr-rejected', 'retr']
        else:
            INFERENCE_TYPE = ['none', 'system'] if use_persona else ['system', 'retr-rejected', 'retr', 'oracle', 'oracle-rejected']

        INFERENCE_TYPE = ['system']
        
        for inference_type in INFERENCE_TYPE:

            print("\nRunning with:", use_persona, response_type, inference_type, "\n")
            
            def create_exemplar(ex):
                if use_persona:
                    return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
Response: {ex['response'][idx]}""" for idx in range(len(ex['prompt']))]}
                else:
                    return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Response: {ex['response'][idx]}""" for idx in range(len(ex['prompt']))]}
            
            def create_exemplar_test(ex):
                response_prefix = [f"Response: {ex['prompt'][idx].title()} sounds like" if config.params['dataset_name'] == 'Mnemonic' else "Response:" for idx in range(len(ex['prompt']))]
                if use_persona:
                    return {'prompt': [f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
{response_prefix[idx]}""" for idx in range(len(ex['prompt']))]}
                else:
                    return {'prompt': [f"""Prompt: {ex['prompt'][idx]} {ex['persona'][idx]}
{response_prefix[idx]}""" for idx in range(len(ex['prompt']))]}
            
            train, test = load_few_shot_data(response_type, inference_type)
            train = train.map(create_exemplar, batched=True)
            test = test.map(create_exemplar_test, batched=True)
            
            base_prompt = '\n\n'.join(train['prompt'])
            inf_prompts = [base_prompt + '\n\n' + p for p in test['prompt']]
            
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
            
            fewshot_results_dir = config.params['fewshot_results_dir']
            fewshot_results_dir = f'{fewshot_results_dir}_{use_persona}_{inference_type}_{response_type}.jsonl'
            
            path = '/'.join(fewshot_results_dir.split('/')[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
            
            outputs = []
            stop_token = '\nPrompt:'
            for idx in tqdm.tqdm(range(len(inf_prompts))):
                prompt = inf_prompts[idx]
            
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:], prompt_len=input_ids.shape[1])])
                input_ids = input_ids.to('cuda')
                out = fewshot_model.generate(input_ids, max_new_tokens=512, do_sample=False, stopping_criteria=stopping_criteria).to('cpu').detach()
                out = out[:, input_ids.shape[1]:]
                out = tokenizer.batch_decode(out)
                
                outputs.append({'response': out[0], 'prompt': prompt})
                with open(fewshot_results_dir, 'w') as fout:
                    json.dump(outputs, fout)