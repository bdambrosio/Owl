import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from promptrix.promptrixTypes import PromptFunctions, PromptMemory, PromptSection, Tokenizer
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.AssistantMessage import AssistantMessage

from alphawave.alphawaveTypes import PromptCompletionClient, PromptCompletionOptions, PromptResponse
from alphawave.internalTypes import ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse, CreateCompletionRequest, CreateCompletionResponse
from alphawave.Colorize import Colorize
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class OpenAIClientOptions:
    def __init__(self, apiKey=None, organization = None, endpoint = None, logRequests = False):
        self.apiKey = apiKey
        self.organization = organization
        self.endpoint = endpoint
        self.logRequests = logRequests

def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)

class OpenAIClient(PromptCompletionClient):
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'AlphaWave'

    def __init__(self, **kwargs):
        self.options = {'apiKey':None, 'organization':None, 'endpoint':None, 'logRequests':False}
        self.options.update(kwargs)
        if not self.options['apiKey']:
            print("Client created without an 'apiKey'.")
            raise ValueError

        self._session = requests.Session()

    def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        startTime = time.time()
        max_input_tokens = 1024

        #
        ### form prompt messages
        #
        if isinstance(options, dict):
            argoptions = options
            #options = PromptCompletionOptions(completion_type = argoptions['completion_type'], model = argoptions['model'])
            options = PromptCompletionOptions(completion_type = argoptions['completion_type'], model = 'gpt_4')
            update_dataclass(options, **argoptions)
        if hasattr(options, 'max_input_tokens') and getattr(options, 'max_input_tokens') is not None:
            max_input_tokens = int(options.max_input_tokens)
        result = prompt.renderAsMessages(memory, functions, tokenizer, max_input_tokens)
        if result.tooLong:
            return {'status': 'too_long', 'message': f"Prompt length of {result.length} tokens exceeds max_input_tokens of {max_input_tokens}."}

        #
        ### form json request message
        #
        request = self.copyOptionsToRequest(
            CreateChatCompletionRequest(model=options.model, messages=result.output),
            options,
            ['max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user'])
        jsonbody = asdict(request)
        keys = list(jsonbody.keys())
        for key in keys:
            if jsonbody[key] is None:
                del jsonbody[key]
        
        #if 'max_tokens' in request:
        #    request['max_tokens'] = int(request['max_tokens'])
        if self.options['logRequests']:
            print(Colorize.title('CHAT PROMPT:'))
            for msg in result.output:
                if not isinstance(msg, dict):
                    msg = msg.__dict__
                print(Colorize.output(json.dumps(msg, indent=2)), end='')
            print()
            for key in list(jsonbody.keys()):
                if key != 'messages':
                    print(Colorize.value(key, jsonbody[key]), end=',')
            print()

        result = client.chat.completions.create(**jsonbody)

        if self.options['logRequests']:
            print(Colorize.title('CHAT RESPONSE:'))
            print(Colorize.value('duration', time.time() - startTime, 'ms'))
            print(Colorize.value('usage', result.usage))
            print(Colorize.value('result message', result.choices[0].message))
        return {'status': 'success', 'message': str(result.choices[0].message.content)}

    def addRequestHeaders(self, headers: Dict[str, str], options: OpenAIClientOptions):
        headers['Authorization'] = f"Bearer {options['apiKey']}"
        if options['organization']:
            headers['OpenAI-Organization'] = options['organization']

    def copyOptionsToRequest(self, target: Dict[str, Any], src: Any, fields: list) -> Dict[str, Any]:
        for field in fields:
            if hasattr(src, field) and getattr(src, field) is not None:
                setattr(target,field, getattr(src,field))
        return target

    def createChatCompletion(self, request: CreateChatCompletionRequest) -> requests.Response:
        return self.post(request)

