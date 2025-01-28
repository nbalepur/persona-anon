"""Main file for prompt templates"""

from abc import ABC, abstractmethod
from enums import PromptType
from typing import Any

# Abstract base class for implementing prompts
class Prompt(ABC):

    def __init__(self, prompt_file, delim='\n\n'):
        with open(prompt_file, 'r') as f:
            self.base_prompt = f.read()
        self.delim = delim
        f.close()

    @abstractmethod
    def create_inference_prompt(self, **kwargs):
        """Create the inference part of the prompt"""
        pass

    def create_prompt(self, **kwargs):
        """Create the full prompt"""
        return self.base_prompt + self.create_inference_prompt(**kwargs)

# LLM Persona Inference Prompts
class PersonaInference(Prompt):
    def create_inference_prompt(self, prompt, chosen, rejected):
        return f'Prompt: {prompt}\n---\nChosen Response: {chosen}\n---\nRejected Response: {rejected}\n---\nPersona:'

# LLM Persona Evaluation Prompts
class PersonaEvalPrompt(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona):
        return f'Prompt: {prompt}\n---\nResponse 1: {r1}\n---\nResponse 2: {r2}\n---\nPersona: {persona}\n---\nChosen Response:'

# Preference Prompts
class PreferenceNoPersona(Prompt):
    def create_inference_prompt(self, prompt, r1, r2):
        return f'Prompt: {prompt}\n---\nResponse 1: {r1}\n---\nResponse 2: {r2}\n---\nBetter Response:'

class PreferenceWithPersona(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona1, persona2):
        return f'Prompt: {prompt}\n---\nResponse 1: {r1}\n---\nPersona 1: {persona1}\n---\nResponse 2: {r2}\n---\nPersona 2: {persona2}\n---\nBetter Response:'

class PersonaPreference(Prompt):
    def create_inference_prompt(self, persona1, persona2):
        return f'Persona 1: {persona1}\n---\nPersona 2: {persona2}\n---\nBetter Persona:'

class PromptFactory:

    def __init__(self, args: Any, prompt_type: PromptType):

        ds_name_parsed = args.inference_split
        self.prompt_dir = f'{args.prompt_dir}{prompt_type.value}/{ds_name_parsed}.txt'

        # map experiment -> prompt template
        self.prompt_type_map = {
            PromptType.persona_inference: PersonaInference(self.prompt_dir),
            PromptType.persona_accuracy: PersonaEvalPrompt(self.prompt_dir),
            PromptType.persona_prefs: PersonaPreference(self.prompt_dir),
        }
        
    def get_prompt(self, prompt_type) -> Prompt:
        if prompt_type in self.prompt_type_map:
            return self.prompt_type_map[prompt_type]
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")
