from enum import Enum
from transformers import StoppingCriteria

# which personas to train on
class TrainingType(Enum):
    chosen = 'chosen' # Chosen response personas
    rejected = 'rejected' # Rejected response personas
    all = 'all' # Both chosen + rejected personas
    none = 'none' # Don't use any personas (normal model training)

# which persona source to use during inference
class InferenceType(Enum):
    gold_chosen = 'gold-chosen' # Persona inferred from the gold chosen response
    retrieved_chosen = 'retr-chosen' # Chosen response persona retrieved from a similar training example 
    gold_rejected = 'gold-rejected' # Persona inferred from the gold rejected response
    retrieved_rejected = 'retr-rejected' # Rejected response persona retrieved from a similar training example 
    system = 'system' # System prompt (fixed across examples)
    none = 'none' # Don't use a persona

class ModelType(Enum):
    fewshot = 'fewshot' # Few-shot Prompting
    sft = 'sft' # Supervised Fine-Tuning
    dpo = 'dpo' # Direct Preference Optimization

class Persona:
    """Combines enums into a Persona."""
    def __init__(self, training_type: TrainingType, inference_type: InferenceType):
        self.training_type = training_type
        self.inference_type = inference_type

    def __str__(self):
        return f"Persona(training_type={self.training_type.value}, inference_type={self.inference_type.value})"

"""Stopping criteria for generation."""
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens = [], prompt_len = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores):
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist)+i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen