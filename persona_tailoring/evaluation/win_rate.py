from model.util import TrainingType, InferenceType, Persona, ModelType
from model import config

import json
import os
import argparse

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

def main(args):

    # load data
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

    eval_dir = f'{res_dir}/{dataset_name}/{model_nickname}/comparisons/{args.model_type_base}_{persona_base_config}_vs_{args.model_type_test}_{persona_test_config}.json'
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"The evaluation file does not exist: {eval_dir}. Please make sure this file is correctly specified and that you have already run run_judge.py")
    with open(eval_dir, "r") as f:
        all_metric_data = json.load(f)
    
    # iterate through both scores
    for metric_name, metric_data in all_metric_data.items():
        metric_map = dict()

        instr = metric_data['instructions']
        is_swapped = metric_data['is_swapped']
        winner = metric_data['winner']

        for prompt, swap, win in zip(instr, is_swapped, winner):
            arr = metric_map.get(prompt, [])
            arr.append(win if not swap else ('A' if win == 'B' else 'B'))
            metric_map[prompt] = arr

        all_winners = []
        for _, winners in metric_map.items():
            if set(winners) == {'A', 'B'}:
                all_winners.append('Tie')
            else:
                all_winners.append(winners[0])

        print()
        print(f"Metric: {metric_name.title().replace('_', ' ')}")
        for label in ['A', 'Tie', 'B']:
            acc = float(sum([l == label for l in all_winners])) / len(all_winners)
            print(f'{label}: {acc}')
        print()


if __name__ == '__main__':
    args = parse_args()
    main(args)