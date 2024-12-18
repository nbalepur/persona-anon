from enum import Enum

class PromptType(Enum):
    persona_inference = 'persona_inference' # few-shot persona inference
    persona_accuracy = 'persona_accuracy' # few-shot evaluation of persona accuracy
    persona_prefs = 'persona_prefs' # LLM preferences on personas

class ModelType(Enum):
    hf_chat = 'hf_chat' # huggingface
    open_ai = 'open_ai' # OpenAI
    cohere = 'cohere' # Cohere
    anthropic = 'anthropic' # Anthropic