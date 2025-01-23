from abc import ABC, abstractmethod
from enums import PromptType
from typing import Any

# Abstract base class for implementing prompts
class Prompt(ABC):

    def __init__(self, prompt_file, delim='\n\n'):
        f = open(prompt_file, 'r')
        self.base_prompt = f.read()
        self.delim = delim
        f.close()

    @abstractmethod
    def create_inference_prompt(self, prompt, r1, r2, persona=None):
        """Create the inference part of the prompt"""
        pass

    def create_prompt(self, prompt, r1, r2, persona=None):
        """Create the full prompt"""
        return self.base_prompt + self.delim + self.create_inference_prompt(prompt, r1, r2, persona)

# LLM Rationale Prompts
class Vanilla(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona):
        return f'Prompt: {prompt}\n---\nChosen Response: {r1}\n---\nRejected Response: {r2}\n---\nPersona:'

# LLM Persona Evaluation Prompts
class PersonaEvalPrompt(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona):
        return f'Prompt: {prompt}\n---\nResponse 1: {r1}\n---\nResponse 2: {r2}\n---\nPersona: {persona}\n---\nChosen Response:'

# Preference Prompts
class PreferenceNoPersona(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona):
        return f'Prompt: {prompt}\n---\nResponse 1: {r1}\n---\nResponse 2: {r2}\n---\nBetter Response:'

class PreferenceWithPersona(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona):
        p1, p2 = persona
        return f'Prompt: {prompt}\n---\nResponse 1: {r1}\n---\nPersona 1: {p1}\n---\nResponse 2: {r2}\n---\nPersona 2: {p2}\n---\nBetter Response:'

class PersonaPreference(Prompt):
    def create_inference_prompt(self, prompt, r1, r2, persona):
        p1, p2 = persona
        return f'Persona 1: {p1}\n---\nPersona 2: {p2}\n---\nBetter Persona:'

class PromptFactory:

    def __init__(self, args: Any, prompt_type: PromptType):

        ds_name_parsed = args.inference_split
        self.prompt_dir = f'{args.prompt_dir}{prompt_type.value}/{ds_name_parsed}.txt'
        self.prompt_dir_eval = f'{args.prompt_dir}persona_accuracy/{ds_name_parsed}.txt'

        self.prompt_type_map = {
            PromptType.persona_inference: Vanilla(self.prompt_dir),
            PromptType.persona_accuracy: PersonaEvalPrompt(self.prompt_dir_eval),
            PromptType.persona_prefs: PersonaPreference(self.prompt_dir),
        }
        
    def get_prompt(self, prompt_type) -> Prompt:
        if prompt_type in self.prompt_type_map:
            return self.prompt_type_map[prompt_type]
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")
