
from model.util import TrainingType

def create_exemplar_train_with_persona_eot(ex):
    """Create exemplars for training data with personas."""
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
Response: {ex['response'][idx]}<|end_of_text|>""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_train_no_persona_eot(ex):
    """Create exemplars for training data without personas."""
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Response: {ex['response'][idx]}<|end_of_text|>""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_train_with_persona_nores(ex):
    """Create exemplars for training data with personas."""
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
Response:""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_train_no_persona_nores(ex):
    """Create exemplars for training data without personas."""
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Response:""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_train_with_persona(ex):
    """Create exemplars for training data with personas."""
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
Response: {ex['response'][idx]}""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_train_no_persona(ex):
    """Create exemplars for training data without personas."""
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Response: {ex['response'][idx]}""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_test_with_persona(ex):
    """Create exemplars for test data with personas."""
    response_prefix = ["Response:" for _ in range(len(ex['prompt']))]
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
{response_prefix[idx]}""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_test_with_persona_mnemonic(ex):
    """Create exemplars for test data with personas."""
    response_prefix = [f"Response: {ex['prompt'][idx].title()} sounds like" for idx in range(len(ex['prompt']))]
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}
Persona: {ex['persona'][idx]}
{response_prefix[idx]}""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_test_no_persona(ex):
    """Create exemplars for test data without personas."""
    response_prefix = ["Response:" for _ in range(len(ex['prompt']))]
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]}{ex['persona'][idx].replace('The user is', ' I am')}
{response_prefix[idx]}""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def create_exemplar_test_no_persona_mnemonic(ex):
    """Create exemplars for test data without personas."""
    response_prefix = [f"Response: {ex['prompt'][idx].title()} sounds like" for idx in range(len(ex['prompt']))]
    return {
        'prompt': [
            f"""Prompt: {ex['prompt'][idx]} {ex['persona'][idx].replace('The user is', 'I am')}
{response_prefix[idx]}""" 
            for idx in range(len(ex['prompt']))
        ]
    }

def fetch_training_template(training_type, add_eot, add_response):
    if not add_response:
        return create_exemplar_train_no_persona_nores if training_type == TrainingType.none else create_exemplar_train_with_persona_nores
    if add_eot:
        return create_exemplar_train_no_persona_eot if training_type == TrainingType.none else create_exemplar_train_with_persona_eot
    else:
        return create_exemplar_train_no_persona if training_type == TrainingType.none else create_exemplar_train_with_persona

def fetch_testing_template(training_type, dataset_split):
    if training_type == TrainingType.none:
        return create_exemplar_test_no_persona_mnemonic if dataset_split == 'Mnemonic' else create_exemplar_test_no_persona
    else:
        return create_exemplar_test_with_persona_mnemonic if dataset_split == 'Mnemonic' else create_exemplar_test_with_persona