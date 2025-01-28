"""
Helper functions for constructing prompt templates for model training/inference
"""

from model.util import TrainingType, InferenceType, ModelType

def create_train_exemplar(ex, use_persona: bool, add_eot: bool, add_response: bool):
    """Create prompt templates for training"""
    prompts = []
    for idx in range(len(ex['prompt'])):

        # included in every prompt
        prompt = f"Prompt: {ex['prompt'][idx]}"
        if use_persona:
            prompt += f"\nPersona: {ex['persona'][idx]}"
        prompt += "\nResponse:"
        
        if add_response: # DPO doesn't need a response (since it uses chosen/rejected), but SFT/Few-shot use the whole sequence
            prompt += f" {ex['response'][idx]}"
        if add_eot: # SFT needs a forced EOT token
            prompt += "<|end_of_text|>"

        prompts.append(prompt)

    return {'prompt': prompts}

def create_test_exemplar(ex, use_persona_training: bool, use_persona_inference: bool, is_mnemonic: bool):
    """Create exemplars for the prompt template"""
    prompts = []
    for idx in range(len(ex['prompt'])):

        # input prompt
        prompt = f"Prompt: {ex['prompt'][idx]}"

        # input persona
        if use_persona_inference:
            if use_persona_training: # if we trained on personas, add the special tag
                prompt += f"\nPersona: {ex['persona'][idx]}"
            else: # if we didn't train on personas, the model must generalize
                prompt += (" " + ex['persona'][idx].replace("The user is", "I am"))
        
        # add the response (and special decoding for Mnemonic)
        prompt += "\nResponse:"
        if is_mnemonic:
            prompt += f" {ex['prompt'][idx].title()} sounds like"

    return {'prompt': prompts}

def fetch_training_template(training_type: TrainingType, model_type: ModelType):
    """Fetch the right prompt template for training. DPO doesn't need the response, but SFT and Few-Shot do. SFT needs an end-of-text token to enforce termination, but the others do not"""
    return lambda ex: create_train_exemplar(
        ex=ex,
        use_persona=(training_type != TrainingType.none),
        add_eot=(model_type == ModelType.sft),
        add_response=(model_type == ModelType.dpo)
    )

def fetch_testing_template(training_type: TrainingType, inference_type: InferenceType, dataset_split: str):
    """Fetch the right prompt template for inference"""
    return lambda ex: create_test_exemplar(
        ex=ex,
        use_persona_training=(training_type != TrainingType.none),
        use_persona_inference=(inference_type != InferenceType.none),
        is_mnemonic=(dataset_split == 'Mnemonic'),
    )