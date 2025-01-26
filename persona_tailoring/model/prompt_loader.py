
from model.util import TrainingType

def create_exemplars(ex, use_persona: bool, add_eot: bool, add_response: bool, is_mnemonic: bool, is_train: bool):
    """Create exemplars for the prompt template"""
    prompts = []
    for idx in range(len(ex['prompt'])):

        # included in every prompt
        prompt = f"Prompt: {ex['prompt'][idx]}"
        if use_persona:
            prompt += f"\nPersona: {ex['persona'][idx]}"
        prompt += "\nResponse:"

        if is_train:
            if add_response: # DPO doesn't need a response
                prompt += f" {ex['response'][idx]}"
            if add_eot: # SFT needs a forced EOT token
                prompt += "<|end_of_text|>"
        else:
            # if we're in inference and on the mnemonic dataset, add the special decoding prefix
            if is_mnemonic:
                prompt += f" {ex['prompt'][idx].title()} sounds like"

        prompts.append(prompt)
    return {'prompt': prompts}

def fetch_training_template(training_type: TrainingType, add_eot: bool, add_response: bool):
    """Fetch the right prompt template for training. DPO doesn't need the response, but SFT and Few-Shot do. SFT needs an end-of-text token to enforce termination, but the others do not"""
    return lambda ex: create_exemplars(
        ex=ex,
        use_persona=(training_type != TrainingType.none),
        add_eot=add_eot,
        add_response=add_response,
        is_mnemonic=False,
        is_train=True,
    )

def fetch_testing_template(training_type: TrainingType, dataset_split: str):
    """Fetch the right prompt template for inference"""
    return lambda ex: create_exemplars(
        ex=ex,
        use_persona=(training_type != TrainingType.none),
        add_eot=False,
        add_response=True,
        is_mnemonic=(dataset_split == 'Mnemonic'),
        is_train=False,
    )