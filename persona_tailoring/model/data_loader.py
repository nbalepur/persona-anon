import datasets
from model import config
from model.util import TrainingType, Persona

class DataLoader:
    """A class to manage loading and preparing datasets."""

    SYSTEM_PERSONA_MAP = { # add/modify any system prompts you want to experiment with here
        'BeaverTails': "The user is meticulous and prefers responses that cover multiple, diverse angles.",
        'Mnemonic': "The user prefers indirect, step-by-step mnemonics that capture the essence of the vocabulary term.",
        'Anthropic HHH': "The user is solution-focused, results-oriented, and fact-oriented, and prefers responses that cover varied angles."
    }
    RESPONSE_TYPE_MAP = {
        TrainingType.none: ['chosen'],
        TrainingType.chosen: ['chosen'],
        TrainingType.rejected: ['rejected'],
        TrainingType.all: ['chosen', 'rejected'],
    }
    
    def __init__(self, dataset_split, dataset_name='nbalepur/persona-tailoring'):
        self.dataset_name = dataset_name
        self.cache_dir = config.params['cache_dir']
        self.system_persona_map = self.SYSTEM_PERSONA_MAP
        self.response_type_map = self.RESPONSE_TYPE_MAP
        self.dataset = datasets.load_dataset(dataset_name, dataset_split, cache_dir=self.cache_dir)
        self.dataset_split = dataset_split
    
    def _get_system_persona(self):
        """Retrieve the system persona for the dataset."""
        return self.system_persona_map.get(self.dataset_split, "")
    
    def load_few_shot_data(self, persona: Persona):
        """Load few-shot training and test data."""
        system_persona = self._get_system_persona()
        train_response_type = persona.training_type
        inference_persona_source, inference_response_type = persona.inference_type.value.split('-')[0], 'rejected' if 'rejected' in persona.inference_type.value else 'chosen'

        ds_train = self.dataset['fewshot']
        ds_test = self.dataset['test']

        train_data = {'prompt': [], 'response': [], 'persona': []}
        response_cols = self.response_type_map[train_response_type]
        for r in response_cols:
            train_data['prompt'] += ds_train['prompt']
            train_data['persona'] += ds_train[f'{r}_persona_gold']
            train_data['response'] += ds_train[r]

        test_personas = (
            [system_persona if inference_persona_source == 'system' else '' for _ in ds_test['prompt']]
            if inference_persona_source in {'system', 'none'} 
            else ds_test[f'{inference_response_type}_persona_{inference_persona_source}']
        )
        test_data = {
            'prompt': ds_test['prompt'],
            'response': ['' for _ in ds_test['prompt']],
            'persona': test_personas
        }
        
        return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)
    
    def load_sft_data(self, training_type: TrainingType):
        """Load data for supervised fine-tuning (SFT)."""
        response_type = training_type
        ds_train = self.dataset['sft_train']
        ds_test = self.dataset['sft_val']

        train_data = {'prompt': [], 'response': [], 'persona': []}    
        val_data = {'prompt': [], 'response': [], 'persona': []}  
        for r in self.response_type_map[response_type]:
            train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
            train_data['persona'] = train_data['persona'] + ds_train[f'{r}_persona_gold']
            train_data['response'] = train_data['response'] + ds_train[r]

            val_data['prompt'] = val_data['prompt'] + ds_test['prompt']
            val_data['persona'] = val_data['persona'] + ds_test[f'{r}_persona_gold']
            val_data['response'] = val_data['response'] + ds_test[r]
        
        return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(val_data)
    
    def load_dpo_data(self, training_type: TrainingType):
        """Load data for direct preference optimization (DPO)."""
        response_type = training_type
        ds_train = self.dataset['dpo_train']
        ds_test = self.dataset['dpo_val']

        train_data = {'prompt': [], 'chosen': [], 'rejected': [], 'persona': []}    
        val_data = {'prompt': [], 'chosen': [], 'rejected': [], 'persona': []}  
        for r in self.response_type_map[response_type]:
            train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
            train_data['persona'] = train_data['persona'] + ds_train[f'{r}_persona_gold']
            train_data['chosen'] = train_data['chosen'] + ds_train[r]
            train_data['rejected'] = train_data['rejected'] + ds_train['rejected' if r == 'chosen' else 'chosen']

            val_data['prompt'] = val_data['prompt'] + ds_test['prompt']
            val_data['persona'] = val_data['persona'] + ds_test[f'{r}_persona_gold']
            val_data['chosen'] = val_data['chosen'] + ds_test[r]
            val_data['rejected'] = val_data['rejected'] + ds_test['rejected' if r == 'chosen' else 'chosen']
        
        return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(val_data)


    def load_test_data(self, persona: Persona):
        """Load test data for evaluation."""
        system_persona = self._get_system_persona()
        inference_persona_source, inference_response_type = persona.inference_type.value.split('-')[0], 'rejected' if 'rejected' in persona.inference_type.value else 'chosen'
        
        ds_test = self.dataset['test']
        test_personas = (
            [system_persona if inference_persona_source == 'system' else '' for _ in ds_test['prompt']]
            if inference_persona_source in {'system', 'none'} 
            else ds_test[f'{inference_response_type}_persona_{inference_persona_source}']
        )

        test_data = {'prompt': ds_test['prompt'], 'response': ['' for _ in ds_test['prompt']], 'persona': test_personas}
        return datasets.Dataset.from_dict(test_data)
