from enums import PromptType
import json
import os
import datasets
from prompt import PromptFactory
import re
from pathlib import Path
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from abc import ABC, abstractmethod

class DataFetcher(ABC):
    @abstractmethod
    def get_data(self):
        """Retrieve data from the source."""
        pass

class PromptResponseFetcher(DataFetcher):

    def __init__(self, ds_name, split_name):
        self.ds = self.load_hf_dataset(ds_name, split_name)

    def load_hf_dataset(self, ds_name, split_name):
        if os.path.isfile(ds_name):
            ds = datasets.load_from_disk(ds_name, split_name)
        else:
            ds = datasets.load_dataset(ds_name, split_name)
        return ds['train']
            
    def get_data(self):
        prompts, r1, r2 = self.ds['prompt'], self.ds['chosen'], self.ds['rejected']
        all_prompts, all_r1, all_r2 = prompts + prompts, r1 + r2, r2 + r1
        return list(zip(all_prompts, all_r1, all_r2))

class ReasoningFetcher(DataFetcher):

    def __init__(self, ds_name, split_name, run_name, res_dir, prompt_type, evalulator_model_name):
        
        # obtain results from all models that have been run on the same split/run_name/prompt_type
        valid_files = []
        required_substrings = [split_name, run_name, 'persona_inference.jsonl']
        directory = Path(res_dir)
        for file in directory.rglob('*'):
            if file.is_file():
                valid = True
                for substr in required_substrings:
                    if substr.strip() not in str(file):
                        valid = False
                        break
                if valid:
                    valid_files.append(file)

        personas = []
        prompts = []
        r1 = []
        r2 = []

        gold_data = []
        num_skipped = 0

        for file in valid_files:
            file = str(file)
            model_name = file[file.index(res_dir) + len(res_dir):]
            model_name = model_name.split('/')[0].strip()
            with open(file, 'r') as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    json_data = json.loads(line)
                    persona = json_data['raw_text']
                    if persona == None: 
                        num_skipped += 1
                        continue
                    match = re.match(r"Prompt:\s*(?P<p>.*)\n---\nChosen Response:\s*(?P<c>.*)\n---\nRejected Response:\s*(?P<r>.*)\n---\nPersona:", json_data['prompt'], re.DOTALL)
                    if match:
                        p = match.group('p').strip()
                        c = match.group('c').strip()
                        r = match.group('r').strip()
                        for order in [(c, r, 1), (r, c, 2)]:
                            r1_, r2_, l = order
                            prompts.append(p)
                            r1.append(r1_)
                            r2.append(r2_)
                            personas.append(persona)

                            gold_data.append({'model_name': model_name, 'label': l, 'is_chosen': line_idx < len(lines) // 2})

        with open(f'{res_dir}{evalulator_model_name}/{split_name}/{run_name}/persona_accuracy_key.jsonl', 'w+') as handle:
            for output in gold_data:
                json.dump(output, handle)
                handle.write('\n')

        self.data = list(zip(prompts, r1, r2, personas))

    def get_data(self):
        return self.data

class PreferencePersonasFetcher(DataFetcher):


    def __init__(self, ds_name, split_name, run_name, res_dir, prompt_type, evalulator_model_name):
        
        # obtain results from all models that have been run on the same split/run_name/prompt_type
        valid_files = [f'{res_dir}/{evalulator_model_name}/{split_name}/{run_name}/persona_inference.jsonl']

        all_data = dict()
        prompt_order = []

        for file in valid_files:
            file = str(file)
            model_name = file[file.index(res_dir) + len(res_dir):]
            model_name = model_name.split('/')[0].strip()
            with open(file, 'r') as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    json_data = json.loads(line)
                    persona = json_data['raw_text']
                    if persona == None: 
                        continue
                    persona = persona.strip().split('\n')[0].strip()
                    match = re.match(r"Prompt:\s*(?P<p>.*)\n---\nChosen Response:\s*(?P<c>.*)\n---\nRejected Response:\s*(?P<r>.*)\n---\nPersona:", json_data['prompt'], re.DOTALL)
                    if match:
                        p = match.group('p').strip()
                        c = match.group('c').strip()
                        r = match.group('r').strip()

                        if p not in prompt_order:
                            prompt_order.append(p)

                        arr = all_data.get(p, {'r1': None, 'p1': None, 'r2': None, 'p2': None})
                        if arr['r1'] == None:
                            arr['r1'] = c
                            arr['p1'] = persona
                        else:
                            arr['r2'] = c
                            arr['p2'] = persona
                        all_data[p] = arr

        prompts = []
        r1 = []
        r2 = []
        personas = []

        for p in prompt_order:
            arr = all_data[p]

            prompts.append(p)
            r1.append(arr['r1'])
            r2.append(arr['r2'])
            personas.append((arr['p1'], arr['p2']))

            prompts.append(p)
            r1.append(arr['r2'])
            r2.append(arr['r1'])
            personas.append((arr['p2'], arr['p1']))

        self.data = list(zip(prompts, r1, r2, personas))

    def get_data(self):
        return self.data

class DataFetcherFactory:

    @staticmethod
    def get_data_fetcher(prompt_type, args, checkpoint_loader):
        if prompt_type in {PromptType.persona_inference}:
            return PromptResponseFetcher(args.dataset_name, args.inference_split)
        elif prompt_type in {PromptType.persona_accuracy}:
            return ReasoningFetcher(args.dataset_name, args.inference_split, args.run_name, args.res_dir, prompt_type, args.model_name)
        elif prompt_type in {PromptType.persona_prefs}:
            return PreferencePersonasFetcher(args.dataset_name, args.inference_split, args.run_name, args.res_dir, prompt_type, args.model_name)
        else:
            raise ValueError(f"Unsupported DataFetcher type: {prompt_type}")

class PromptCollator:

    def __init__(self, args):
        self.data_fetcher_factory = DataFetcherFactory()
        self.args = args

    def get_prompts(self, prompt_type, checkpoint_loader):
        
        data_fetcher = self.data_fetcher_factory.get_data_fetcher(prompt_type, self.args, checkpoint_loader)

        prompt_factory = PromptFactory(self.args, prompt_type)
        prompt_parser = prompt_factory.get_prompt(prompt_type)

        prompt_inputs = data_fetcher.get_data()
        prompts = [None if p == None else prompt_parser.create_prompt(*p) for p in prompt_inputs]
        return prompts
        