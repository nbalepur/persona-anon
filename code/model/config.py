params = {
    'model_nickname': 'llama_8b', # the nickname of the model (how you want it to be saved)
    'base_model_name': 'meta-llama/Llama-3.1-8B', # points to the huggingface model as the base model
    
    'use_wandb': True, # should we log the training to wandb?
    'wandb_key': '', # if `use_wandb` is True, put your API key here
    'wandb_name': 'defeasible-alignment', # if `use_wandb` is True, give a name for the project
    'open_ai_key': '...', # OpenAI key for DSPy evaluation

    'load_in_8bit': False, # load the model in 8bit?
    'load_in_4bit': False, # load the model in 4bit?
    
    'hf_read_token': '',

    'cache_dir': '', # cache directory to store models, datasets, etc.
    'model_save_dir': '', # directory for where the models should be saved
    'results_save_dir': '', # directory for where the results should be saved

    'dataset_name': 'Mnemonic', # which dataset to use (Mnemonic, Safe_RLHF, HH_RLHF)
}

# specify the device_map
device_map = 'auto'



# ******************* these parameters are all specified automatically based on what you put above! *******************
params['adj_token_model_name'] = f'{params["model_save_dir"]}/{params["model_nickname"]}-adj'
params['sft_output_dir'] = f'{params["model_save_dir"]}/{params["dataset_name"]}/sft_{params["model_nickname"]}'
params['sft_final_output_dir'] = f'{params["model_save_dir"]}/{params["dataset_name"]}/sft_{params["model_nickname"]}_final'
params['dpo_output_dir'] = f'{params["model_save_dir"]}/{params["dataset_name"]}/dpo_{params["model_nickname"]}'
params['dpo_final_output_dir'] = f'{params["model_save_dir"]}/{params["dataset_name"]}/dpo_{params["model_nickname"]}_final'
params['sft_adapter_name'] = f'{params["model_save_dir"]}/{params["dataset_name"]}/sft_{params["model_nickname"]}_adapter'
params['sft_tokenizer_name'] =  f'{params["model_save_dir"]}/{params["dataset_name"]}/sft_{params["model_nickname"]}_tokenizer'
params['dpo_adapter_name'] = f'{params["model_save_dir"]}/{params["dataset_name"]}/dpo_{params["model_nickname"]}_adapter'
params['fewshot_results_dir'] = f'{params["results_save_dir"]}/{params["dataset_name"]}/{params["model_nickname"]}/fewshot'
params['sft_results_dir'] = f'{params["results_save_dir"]}/{params["dataset_name"]}/{params["model_nickname"]}/sft'
params['dpo_results_dir'] = f'{params["results_save_dir"]}/{params["dataset_name"]}/{params["model_nickname"]}/dpo'