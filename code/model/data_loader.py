import datasets
import config

SYSTEM_PERSONA = "The user is meticulous and prefers responses that cover multiple, diverse angles."
#SYSTEM_PERSONA = "The user prefers indirect, step-by-step mnemonics that capture the essence of the vocabulary term."
#SYSTEM_PERSONA = "The user is solution-focused, results-oriented, and fact-oriented, and prefers responses that cover varied angles."

response_type_map = {
    'chosen': ['chosen'],
    'rejected': ['rejected'],
    'all': ['chosen', 'rejected'],
}

DS_NAME = '' # dataset name
ds = datasets.load_dataset(DS_NAME, cache_dir=config.params['cache_dir'])

def load_few_shot_data_name(response_type, inference_persona_type, ds_name):

    if '-rejected' in inference_persona_type:
        inference_persona_type, inference_response_type = inference_persona_type.replace('-rejected', ''), 'rejected'
    else:
        inference_persona_type, inference_response_type = inference_persona_type, 'chosen'
    
    ds_train, ds_test = ds[f'{ds_name}_fewshot'], ds[f'{ds_name}_test']

    train_data = {'prompt': [], 'response': [], 'persona': []}
    
    test_data = {'prompt': ds_test['prompt'], 'persona': [(SYSTEM_PERSONA if inference_persona_type == 'system' else '') for _ in ds_test['prompt']] if inference_persona_type in {'system', 'none'} else ds_test[f'llama_{inference_response_type}_persona_{inference_persona_type}']}
    
    response_cols = response_type_map[response_type]
    for r in response_cols:
        train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
        train_data['persona'] = train_data['persona'] + ds_train[f'llama_{r}_persona_oracle']
        train_data['response'] = train_data['response'] + ds_train[r]
        
    return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)

def load_few_shot_data(response_type, inference_persona_type):

    if '-rejected' in inference_persona_type:
        inference_persona_type, inference_response_type = inference_persona_type.replace('-rejected', ''), 'rejected'
    else:
        inference_persona_type, inference_response_type = inference_persona_type, 'chosen'
    
    ds_name = config.params['dataset_name']
    ds_train, ds_test = ds[f'{ds_name}_fewshot'], ds[f'{ds_name}_test']

    train_data = {'prompt': [], 'response': [], 'persona': []}

    
    test_data = {'prompt': ds_test['prompt'], 'persona': [(SYSTEM_PERSONA if inference_persona_type == 'system' else '') for _ in ds_test['prompt']] if inference_persona_type in {'system', 'none'} else ds_test[f'llama_{inference_response_type}_persona_{inference_persona_type}']}
    
    response_cols = response_type_map[response_type]
    for r in response_cols:
        train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
        train_data['persona'] = train_data['persona'] + ds_train[f'llama_{r}_persona_oracle']
        train_data['response'] = train_data['response'] + ds_train[r]
        
    return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)

def load_sft_data(use_persona, response_type):
    
    ds_name = config.params['dataset_name']
    ds_train, ds_test = ds[f'{ds_name}_sft_train'], ds[f'{ds_name}_sft_val']

    train_data = {'prompt': [], 'response': [], 'persona': []}    
    test_data = {'prompt': [], 'response': [], 'persona': []}    
    response_cols = response_type_map[response_type]
    
    for r in response_cols:
        train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
        train_data['persona'] = train_data['persona'] + ds_train[f'llama_{r}_persona_oracle']
        train_data['response'] = train_data['response'] + ds_train[r]

        test_data['prompt'] = test_data['prompt'] + ds_test['prompt']
        test_data['persona'] = test_data['persona'] + ds_test[f'llama_{r}_persona_oracle']
        test_data['response'] = test_data['response'] + ds_test[r]
        
    return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)

def load_sft_data_name(use_persona, response_type, ds_name):
    
    ds_train, ds_test = ds[f'{ds_name}_sft_train'], ds[f'{ds_name}_sft_val']

    train_data = {'prompt': [], 'response': [], 'persona': []}    
    test_data = {'prompt': [], 'response': [], 'persona': []}    
    response_cols = response_type_map[response_type]
    
    for r in response_cols:
        train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
        train_data['persona'] = train_data['persona'] + ds_train[f'llama_{r}_persona_oracle']
        train_data['response'] = train_data['response'] + ds_train[r]

        test_data['prompt'] = test_data['prompt'] + ds_test['prompt']
        test_data['persona'] = test_data['persona'] + ds_test[f'llama_{r}_persona_oracle']
        test_data['response'] = test_data['response'] + ds_test[r]
        
    return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)

def load_dpo_data(use_persona, response_type):

    ds_name = config.params['dataset_name']
    ds_train, ds_test = ds[f'{ds_name}_dpo_train'], ds[f'{ds_name}_dpo_val']

    train_data = {'prompt': [], 'chosen': [], 'rejected': [], 'persona': []}    
    test_data = {'prompt': [], 'chosen': [], 'rejected': [], 'persona': []}    
    response_cols = response_type_map[response_type]
    
    for r in response_cols:
        train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
        train_data['persona'] = train_data['persona'] + ds_train[f'llama_{r}_persona_oracle']
        train_data['chosen'] = train_data['chosen'] + ds_train[r]
        train_data['rejected'] = train_data['rejected'] + ds_train['rejected' if r == 'chosen' else 'chosen']

        test_data['prompt'] = test_data['prompt'] + ds_test['prompt']
        test_data['persona'] = test_data['persona'] + ds_test[f'llama_{r}_persona_oracle']
        test_data['chosen'] = test_data['chosen'] + ds_test[r]
        test_data['rejected'] = test_data['rejected'] + ds_test['rejected' if r == 'chosen' else 'chosen']
        
    return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)

def load_dpo_data_name(use_persona, response_type, ds_name):

    ds_train, ds_test = ds[f'{ds_name}_dpo_train'], ds[f'{ds_name}_dpo_val']

    train_data = {'prompt': [], 'chosen': [], 'rejected': [], 'persona': []}    
    test_data = {'prompt': [], 'chosen': [], 'rejected': [], 'persona': []}    
    response_cols = response_type_map[response_type]
    
    for r in response_cols:
        train_data['prompt'] = train_data['prompt'] + ds_train['prompt']
        train_data['persona'] = train_data['persona'] + ds_train[f'llama_{r}_persona_oracle']
        train_data['chosen'] = train_data['chosen'] + ds_train[r]
        train_data['rejected'] = train_data['rejected'] + ds_train['rejected' if r == 'chosen' else 'chosen']

        test_data['prompt'] = test_data['prompt'] + ds_test['prompt']
        test_data['persona'] = test_data['persona'] + ds_test[f'llama_{r}_persona_oracle']
        test_data['chosen'] = test_data['chosen'] + ds_test[r]
        test_data['rejected'] = test_data['rejected'] + ds_test['rejected' if r == 'chosen' else 'chosen']
        
    return datasets.Dataset.from_dict(train_data), datasets.Dataset.from_dict(test_data)

def load_test_data(use_persona, response_type, inference_persona_type):
    ds_name = config.params['dataset_name']

    if '-rejected' in inference_persona_type:
        inference_persona_type, inference_response_type = inference_persona_type.replace('-rejected', ''), 'rejected'
    else:
        inference_persona_type, inference_response_type = inference_persona_type, 'chosen'
    
    ds_test = ds[f'{ds_name}_test']    
    test_data = {'prompt': ds_test['prompt'], 'persona': [(SYSTEM_PERSONA if inference_persona_type == 'system' else '') for _ in ds_test['prompt']] if inference_persona_type in {'system', 'none'} else ds_test[f'llama_{inference_response_type}_persona_{inference_persona_type}']}
    return datasets.Dataset.from_dict(test_data)

def load_test_data_name(use_persona, response_type, inference_persona_type, ds_name):
    ds_name = config.params['dataset_name']

    if '-rejected' in inference_persona_type:
        inference_persona_type, inference_response_type = inference_persona_type.replace('-rejected', ''), 'rejected'
    else:
        inference_persona_type, inference_response_type = inference_persona_type, 'chosen'
    
    ds_test = ds[f'{ds_name}_test']    
    test_data = {'prompt': ds_test['prompt'], 'persona': [(SYSTEM_PERSONA if inference_persona_type == 'system' else '') for _ in ds_test['prompt']] if inference_persona_type in {'system', 'none'} else ds_test[f'llama_{inference_response_type}_persona_{inference_persona_type}']}
    return datasets.Dataset.from_dict(test_data)