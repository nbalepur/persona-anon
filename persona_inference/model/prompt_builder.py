from persona_inference.model.prompt_template_loader import PromptFactory, PromptType
from persona_inference.model.prompt_data_loader import DataFetcherFactory

class PromptBuilder:

    def __init__(self, args):
        self.args = args

    def get_prompts(self, prompt_type: PromptType):

        # template for the prompt
        prompt_factory = PromptFactory(args=self.args, prompt_type=prompt_type)
        prompt_parser = prompt_factory.get_prompt(prompt_type)

        # data inputs used to infill the prompt template
        data_factory = DataFetcherFactory()
        data_fetcher = data_factory.get_data_fetcher(prompt_type=prompt_type, args=self.args)
        prompt_inputs = data_fetcher.get_data()

        prompts = [None if p == None else prompt_parser.create_prompt(**p) for p in prompt_inputs]
        return prompts