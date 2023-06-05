from promptrix.promptrixTypes import PromptFunctions, PromptMemory, Tokenizer
from alphawaveTypes import PromptResponse, Validation, PromptResponseValidator

class DefaultResponseValidator(PromptResponseValidator):
    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts) -> Validation:
        self.feedback = response['message']['content'] if isinstance(response['message'], dict) else response.message
        return {
            'type': 'Validation',
            'valid': True,
            'value': response['message']['content'] if isinstance(response['message'], dict) else response.message
        }
