import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pickle
import pandas as pd
import datasets
import numpy as np
import json
import tqdm
import re
import nltk
from model import config

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF, SCORE_RUBRIC_TEMPLATE

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", enforce_eager=True, gpu_memory_utilization=0.9)
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT_WO_REF)


def parse_json(f):
    with open(f, 'r') as json_file:
        json_list = list(json_file)
    outputs = []
    prompts = []
    personas = []
    for json_str in json_list:
        result = json.loads(json_str)
        for r in result:
            outputs.append(r['response'])

            if 'Persona:' in r['prompt']:
                p = r['prompt'].split('\nPrompt:')[-1]
                prompt = p[:p.index("Persona:")].strip()
                persona = p[p.index("Persona: ") + len("Persona: "):p.index("Response:")].strip()
            else:
                p = r['prompt'].split('\nPrompt:')[-1]
                persona = ""
                prompt = p[:p.index("Response:")].strip()
            prompts.append(prompt)
            personas.append(persona)
    return prompts, personas, outputs

# ********************* Main method *********************

instruction_map = {
    'Safe_RLHF': 'Please provide a response to the following prompt.',
    'HH_RLHF': 'Please provide a response to the following prompt.',
    'Mnemonic': 'Please provide a keyword mnemonic to the following vocabulary term.',
}

persona_map = {
    'Safe_RLHF': 'Please provide a response that aligns with the attributes of the following persona.',
    'HH_RLHF': 'Please provide a response that aligns with the attributes of the following persona.',
    'Mnemonic': 'Please provide a response that aligns with the attributes of the following persona.'
}

def main():

    all_quality_instructions, all_quality_responses_from_a, all_quality_responses_from_b, all_quality_labels = [], [], [], []
    all_persona_instructions, all_persona_responses_from_a, all_persona_responses_from_b, all_persona_labels = [], [], [], []

    all_configs_a = []
    all_configs_b = []
    all_datasets = []
    
    for dataset in ['HH_RLHF', 'Mnemonic', 'Safe_RLHF']:
        model_nickname = 'llama_8b'
    
        all_base = []
        all_test = []
    
        FEWSHOT_BASE_CONFIG_PERSONAEXP = ('fewshot', False, 'system', 'chosen')
        FEWSHOT_TEST_CONFIGS_PERSONAEXP = []
        for response_type in ['chosen', 'rejected', 'all']:
            FEWSHOT_TEST_CONFIGS_PERSONAEXP += [
                ('fewshot', True, 'system', response_type),
                ('fewshot', True, 'oracle-rejected', response_type),
                ('fewshot', True, 'retr-rejected', response_type),
                ('fewshot', True, 'oracle', response_type),
                ('fewshot', True, 'retr', response_type)
            ]
        all_base.append(FEWSHOT_BASE_CONFIG_PERSONAEXP)
        all_test.append(FEWSHOT_TEST_CONFIGS_PERSONAEXP)
        
        SFT_BASE_CONFIG_PERSONAEXP = ('sft', False, 'system', 'chosen')
        SFT_TEST_CONFIGS_PERSONAEXP = []
        for response_type in ['chosen', 'rejected', 'all']:
            SFT_TEST_CONFIGS_PERSONAEXP += [
                ('sft', True, 'system', response_type),
                ('sft', True, 'oracle-rejected', response_type),
                ('sft', True, 'retr-rejected', response_type),
                ('sft', True, 'oracle', response_type),
                ('sft', True, 'retr', response_type)
            ]
        all_base.append(SFT_BASE_CONFIG_PERSONAEXP)
        all_test.append(SFT_TEST_CONFIGS_PERSONAEXP)
        
        DPO_BASE_CONFIG_PERSONAEXP = ('dpo', False, 'system', 'chosen')
        DPO_TEST_CONFIGS_PERSONAEXP = []
        for response_type in ['chosen', 'rejected', 'all']:
            DPO_TEST_CONFIGS_PERSONAEXP += [
                ('dpo', True, 'system', response_type),
                ('dpo', True, 'oracle-rejected', response_type),
                ('dpo', True, 'retr-rejected', response_type),
                ('dpo', True, 'oracle', response_type),
                ('dpo', True, 'retr', response_type)
            ]
        all_base.append(DPO_BASE_CONFIG_PERSONAEXP)
        all_test.append(DPO_TEST_CONFIGS_PERSONAEXP)

        ABLATION_BASE_CONFIG = ('fewshot', False, 'system', 'chosen')
        ABLATION_TEST_CONFIGS = [('sft', False, 'system', 'chosen'), ('dpo', False, 'system', 'chosen')]
        all_base.append(ABLATION_BASE_CONFIG)
        all_test.append(ABLATION_TEST_CONFIGS)
        
        PERSONA_ABLATION_BASE_CONFIG = ('fewshot', True, 'oracle', 'chosen')
        PERSONA_ABLATION_TEST_CONFIGS = [('sft', True, 'oracle', 'chosen'), ('dpo', True, 'oracle', 'chosen')]
        all_base.append(PERSONA_ABLATION_BASE_CONFIG)
        all_test.append(PERSONA_ABLATION_TEST_CONFIGS)

        PERSONA_ABLATION_BASE_CONFIG_RETR = ('fewshot', True, 'retr', 'chosen')
        PERSONA_ABLATION_TEST_CONFIGS_RETR = [('sft', True, 'retr', 'chosen'), ('dpo', True, 'retr', 'chosen')]
        all_base.append(PERSONA_ABLATION_BASE_CONFIG_RETR)
        all_test.append(PERSONA_ABLATION_TEST_CONFIGS_RETR)

        PERSONA_ABLATION_BASE_CONFIG = ('fewshot', False, 'system', 'chosen')
        PERSONA_ABLATION_TEST_CONFIGS = [('fewshot', True, 'oracle', 'chosen'), ('sft', True, 'oracle', 'chosen'), ('dpo', True, 'oracle', 'chosen')]
        all_base.append(PERSONA_ABLATION_BASE_CONFIG)
        all_test.append(PERSONA_ABLATION_TEST_CONFIGS)

        PERSONA_ABLATION_BASE_CONFIG_RETR = ('fewshot', False, 'system', 'chosen')
        PERSONA_ABLATION_TEST_CONFIGS_RETR = [('fewshot', True, 'retr', 'chosen'), ('sft', True, 'retr', 'chosen'), ('dpo', True, 'retr', 'chosen')]
        all_base.append(PERSONA_ABLATION_BASE_CONFIG_RETR)
        all_test.append(PERSONA_ABLATION_TEST_CONFIGS_RETR)

        for base_config, test_configs in zip(all_base, all_test):
            for config in test_configs:
                
                if dataset == 'Mnemonic' and ('oracle' in base_config[2] or 'oracle' in config[2]):
                    continue
        
                dir_base = f'/sensei-fs-3/users/nbalepur/defeasible-alignment/Mnemonic/results/{dataset}/{model_nickname}/{base_config[0]}_{base_config[1]}_{base_config[2]}_{base_config[3]}.jsonl'
                dir_test = f'/sensei-fs-3/users/nbalepur/defeasible-alignment/Mnemonic/results/{dataset}/{model_nickname}/{config[0]}_{config[1]}_{config[2]}_{config[3]}.jsonl'
                
                p_base, _, r_base = parse_json(dir_base)
                p_test, persona_test, r_test = parse_json(dir_test)

                if dataset == 'Mnemonic':
                    p_base = [x + ' sounds like' for x in p_base]
                    p_test = [x + ' sounds like' for x in p_test]

                all_datasets.extend([dataset for _ in p_base] + [dataset for _ in p_base])
        
                # quality evaluation accumulation
                quality_rubric = "Is the response high-quality?"
                for idx in range(len(p_base)):
                    all_quality_instructions.append(instruction_map[dataset] + ' Prompt: ' + p_base[idx])
                    all_quality_responses_from_a.append(r_base[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_quality_responses_from_b.append(r_test[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_quality_labels.append('B')
                    all_quality_instructions.append(instruction_map[dataset] + ' Prompt: ' + p_base[idx])
                    all_quality_responses_from_a.append(r_test[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_quality_responses_from_b.append(r_base[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_quality_labels.append('A')

                    all_configs_a.append(base_config)
                    all_configs_a.append(config)
                                        
                    all_configs_b.append(config)
                    all_configs_b.append(base_config)

                # persona evaluation accumulation
                persona_rubric = "Does the response align with the user's specified persona?"
                for idx in range(len(p_base)):
                    all_persona_instructions.append(persona_map[dataset] + ' Persona: ' + persona_test[idx])
                    all_persona_responses_from_a.append(r_base[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_persona_responses_from_b.append(r_test[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_persona_labels.append('B')
                    all_persona_instructions.append(persona_map[dataset] + ' Persona: ' + persona_test[idx])
                    all_persona_responses_from_a.append(r_test[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_persona_responses_from_b.append(r_base[idx].replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
                    all_persona_labels.append('A')
        
    # Run evaluations once for quality and once for personalization
    quality_feedback, quality_scores = judge.relative_grade(
        instructions=all_quality_instructions,
        responses_A=all_quality_responses_from_a,
        responses_B=all_quality_responses_from_b,
        rubric=quality_rubric,
        reference_answers=None
    )

    persona_feedback, persona_scores = judge.relative_grade(
        instructions=all_persona_instructions,
        responses_A=all_persona_responses_from_a,
        responses_B=all_persona_responses_from_b,
        rubric=persona_rubric,
        reference_answers=None
    )
    
    # Save results
    eval_res_dir = f'/sensei-fs-3/users/nbalepur/defeasible-alignment/Mnemonic/evaluate/eval_results2/{model_nickname}/'
    os.makedirs(eval_res_dir, exist_ok=True)
    out_dict = {
        'persona': {
            'instructions': all_persona_instructions, 
            'A': all_persona_responses_from_a, 
            'B': all_persona_responses_from_b, 
            'score': persona_scores, 
            'gold': all_persona_labels,
            'config_A': all_configs_a,
            'config_B': all_configs_b,
            'dataset': all_datasets,
        },
        'quality': {
            'instructions': all_quality_instructions, 
            'A': all_quality_responses_from_a, 
            'B': all_quality_responses_from_b, 
            'score': quality_scores, 
            'gold': all_quality_labels,
            'config_A': all_configs_a,
            'config_B': all_configs_b,
            'dataset': all_datasets,
        }
    }
    with open(f'{eval_res_dir}final_eval_results.pkl', 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
if __name__ == '__main__':
    main()