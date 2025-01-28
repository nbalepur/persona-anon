"""
Uses Prometheus to judge the Personalization and Quality of responses from two different models
"""

from model.util import TrainingType, InferenceType, Persona, ModelType
from model import config

import json
import os
import argparse
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DPO Inference Script")
    parser.add_argument(
        '--training_type_base',
        type=str,
        required=True,
        choices=[t.value for t in TrainingType],
        help="Type of training for the baseline model (e.g., 'chosen', 'rejected')"
    )
    parser.add_argument(
        '--inference_type_base',
        type=str,
        required=True,
        choices=[t.value for t in InferenceType],
        help="Type of inference for the baseline model (e.g., 'chosen', 'rejected')"
    )
    parser.add_argument(
        '--model_type_base',
        type=str,
        required=True,
        choices=[t.value for t in ModelType],
        help="Type of model for the baseline model (e.g., 'fewshot', 'sft', 'dpo')"
    )
    parser.add_argument(
        '--training_type_test',
        type=str,
        required=True,
        choices=[t.value for t in TrainingType],
        help="Type of training for the tested model (e.g., 'chosen', 'rejected')"
    )
    parser.add_argument(
        '--inference_type_test',
        type=str,
        required=True,
        choices=[t.value for t in InferenceType],
        help="Type of inference for the tested model (e.g., 'chosen', 'rejected')"
    )
    parser.add_argument(
        '--model_type_test',
        type=str,
        required=True,
        choices=[t.value for t in ModelType],
        help="Type of model for the tested model (e.g., 'fewshot', 'sft', 'dpo')"
    )
    return parser.parse_args()

def setup_model():
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", enforce_eager=True, gpu_memory_utilization=0.5, download_dir=config.params['cache_dir'])
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT_WO_REF)
    return judge

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

INSTRUCTION_MAP = {
    'BeaverTails': 'Please provide a response to the following prompt.',
    'Anthropic HHH': 'Please provide a response to the following prompt.',
    'Mnemonic': 'Please provide a keyword mnemonic to the following vocabulary term.',
}

def main(args):

    persona_base_config = Persona(
        training_type=TrainingType(args.training_type_base),
        inference_type=InferenceType(args.inference_type_base)
    )
    persona_test_config = Persona(
        training_type=TrainingType(args.training_type_base),
        inference_type=InferenceType(args.inference_type_base)
    )

    res_dir = config.params['results_save_dir']
    dataset_name = config.params['dataset_name']
    model_nickname = config.params['model_nickname']
    dir_base = f'{res_dir}/{dataset_name}/{model_nickname}/{args.model_type_base}_{persona_base_config}.jsonl'
    dir_test = f'{res_dir}/{dataset_name}/{model_nickname}/{args.model_type_test}_{persona_test_config}.jsonl'

    quality_instr, quality_res_a, quality_res_b, quality_is_swapped = [], [], [], []
    persona_instr, persona_res_a, persona_res_b, persona_is_swapped = [], [], [], []

    p_base, _, r_base = parse_json(dir_base)
    p_test, persona_test, r_test = parse_json(dir_test)

    # response quality evaluation
    quality_rubric = "Is the response high-quality?"
    for idx in range(len(p_base)):
        for A, B, is_swapped in [(r_base[idx], r_test[idx], False), (r_test[idx], r_base[idx], True)]:
            quality_instr.append(INSTRUCTION_MAP[dataset_name] + ' Prompt: ' + p_base[idx])
            quality_res_a.append(A.replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
            quality_res_b.append(B.replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
            quality_is_swapped.append(is_swapped)

    # personalization evaluation
    persona_rubric = "Does the response answer the prompt and align with the user's specified persona?"
    for idx in range(len(p_base)):
        for A, B, is_swapped in [(r_base[idx], r_test[idx], False), (r_test[idx], r_base[idx], True)]:
            persona_instr.append(INSTRUCTION_MAP[dataset_name] + ' Prompt: ' + p_base[idx] + ' Persona: ' + persona_test[idx])
            persona_res_a.append(A.replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
            persona_res_b.append(B.replace('Prompt:', '').replace('<|end_of_text|>', '').strip())
            persona_is_swapped.append(is_swapped)

    judge = setup_model()
        
    # Run evaluations once for quality and once for personalization
    _, quality_scores = judge.relative_grade(
        instructions=quality_instr,
        responses_A=quality_res_a,
        responses_B=quality_res_b,
        rubric=quality_rubric,
        reference_answers=None
    )
    _, persona_scores = judge.relative_grade(
        instructions=persona_instr,
        responses_A=persona_res_a,
        responses_B=persona_res_b,
        rubric=persona_rubric,
        reference_answers=None
    )

    out_data = {
        'response_quality': {
            'instructions': quality_instr,
            'base_response': quality_res_a,
            'test_response': quality_res_b,
            'winner': quality_scores,
            'is_swapped': quality_is_swapped
        },
        'personalization': {
            'instructions': persona_instr,
            'base_response': persona_res_a,
            'test_response': persona_res_b,
            'winner': persona_scores,
            'is_swapped': persona_is_swapped
        }
    }

    eval_dir = f'{res_dir}/{dataset_name}/{model_nickname}/comparisons/{args.model_type_base}_{persona_base_config}_vs_{args.model_type_test}_{persona_test_config}.json'
    eval_folder = os.path.dirname(eval_dir)
    os.makedirs(eval_folder, exist_ok=True)
    with open(eval_dir, "w") as f:
        json.dump(out_data, f)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)