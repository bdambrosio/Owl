import requests, time, copy
import traceback
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

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium"
client = MistralClient(api_key=api_key)

"""
# quick test of Mistral API
messages = [
    ChatMessage(role="user", content="<<SYS>>You are a talented mathematician<</SYS>>Show a derivation of the Taylor series expansion of sin(x) around 0.")
]

# No streaming
chat_response = client.chat(
    model='mistral-small',
    messages=messages,
    temperature=.1,
    max_tokens=100,
    #stream=False,
    safe_mode=False, #default
)
print(f'\nMistral:\n{chat_response}\n\n')
"""

@dataclass
class MistralAIClientOptions:
    def __init__(self, apiKey=None, organization = None, endpoint = None, logRequests = True):
        self.apiKey = apiKey
        self.organization = organization
        self.endpoint = endpoint
        self.logRequests = logRequests
        
def update_dataclass(instance, **kwargs):
    for key, value in kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)

class MistralAIClient(PromptCompletionClient):
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'AlphaWave'

    def __init__(self, **kwargs):
        global client
        self.options = {'apiKey':None, 'organization':None, 'endpoint':None, 'logRequests':True}
        self.options.update(kwargs)
        print(f' MistralAI self.options: {self.options}')
        self.mistralai_client = client
        self.model=model
        if not self.options['apiKey']:
            print("Client created without an 'apiKey'.")
            raise ValueError

        self._session = requests.Session()

    def completePrompt(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, prompt: PromptSection, options: PromptCompletionOptions) -> PromptResponse:
        startTime = time.time()
        max_input_tokens = 1024
        response = ''
        try:
            
            #
            ### form prompt messages
            #
            if isinstance(options, dict):
                argoptions = options
                options = PromptCompletionOptions(completion_type = argoptions['completion_type'], model = 'medium')
                update_dataclass(options, **argoptions)
            if hasattr(options, 'max_input_tokens') and getattr(options, 'max_input_tokens') is not None:
                max_input_tokens = int(options.max_input_tokens)
            result = prompt.renderAsMessages(memory, functions, tokenizer, max_input_tokens)
            if result.tooLong:
                return {'status': 'too_long', 'message': f"Prompt length of {result.length} tokens exceeds max_input_tokens of {max_input_tokens}."}

            request = self.copyOptionsToRequest(
                CreateChatCompletionRequest(model=options.model, messages=result.output),
                options, ['max_tokens', 'temperature', 'top_p', 'stream', 'messages', 'model'])
            jsonbody = asdict(request)
            keys = list(jsonbody.keys())
            for key in keys:
                if jsonbody[key] is None:
                    del jsonbody[key]
        
            #if 'max_tokens' in request:
            #    request['max_tokens'] = int(request['max_tokens'])
            self.options['logRequests'] = True
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

            result = self.createChatCompletion(jsonbody, result.output)
            response = result.choices[0].message.content
            if self.options['logRequests']:
                print(Colorize.title('CHAT RESPONSE:'))
                print(Colorize.value('duration', time.time() - startTime, 'ms'))
                print(Colorize.value('result message', response))
                #print(Colorize.output(response.json()))
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            
        return {'status': 'success', 'message': str(response)}

    def addRequestHeaders(self, headers: Dict[str, str], options: MistralAIClientOptions):
        headers['Authorization'] = f"Bearer {options['apiKey']}"
        if options['organization']:
            headers['MistralAI-Organization'] = options['organization']

    def copyOptionsToRequest(self, target: Dict[str, Any], src: Any, fields: list) -> Dict[str, Any]:
        for field in fields:
            if hasattr(src, field) and getattr(src, field) is not None:
                setattr(target,field, getattr(src,field))
        return target

    def createChatCompletion(self, request, rendered_prompt_msgs) -> requests.Response:
        ### form mistral prompt
        prompt = []
        #print(f'mistral prompt input {rendered_prompt_msgs}')
        #change below to embed SystemMessage in User
        sysMsg = None; first_user_msg = True
        for i, msg in enumerate(rendered_prompt_msgs):
            msg_role = msg['role']
            msg_content = msg['content']
            # check for special case of last message pair user/assistant, can't have assistant as last!
            if i == len(rendered_prompt_msgs) -2:
                if (msg_role == 'user' or msg_role == 'system') and rendered_prompt_msgs[i+1]['role'] == 'assistant':
                    # trying to prime llm with an open-ended final asst msg, append to last user:
                    response_prime = rendered_prompt_msgs[i+1]['content']
                    chatMessage = ChatMessage(role = 'user', content = msg_content+'\n'+response_prime)
                    prompt.append(chatMessage)
                    break

            # check for first msg a 'system':
            if i == 0 and msg["role"] == 'system':
                sysMsg = msg["content"]
                if len(rendered_prompt_msgs) == 1 or rendered_prompt_msgs[1]['role'] == 'assistant':
                    # this is only msg!
                    chatMessage = ChatMessage(role = 'user', content = sysMsg)
                    prompt.append(chatMessage)
                    continue
                else:
                    continue
            if i==1 and msg["role"] == 'user' and sysMsg is not None:
                msg_content = '<<SYS>>'+sysMsg+'<</SYS>>'+msg['content']
                chatMessage = ChatMessage(role = msg["role"], content = msg_content)
                prompt.append(chatMessage)
                continue

            chatMessage = ChatMessage(role = msg["role"], content = msg_content)
            prompt.append(chatMessage)
        temp = .1
        max_t = 50
        for key in request:
            if key == "temperature":
                temp=float(request[key])
            if key == "max_tokens":
                max_t= int(request[key])
            if key == "model":
                model =request[key]
        print(f'\nmistral prompt\n{prompt}\n\n')
        return self.mistralai_client.chat(model=model, messages = prompt, temperature=temp, max_tokens = max_t)

    # mistral prompt use example
    """
    messages = [
        {"role": "user", "content": "2+2"},
        {"role": "assistant", "content": "4!"},
        {"role": "user", "content": "+2"},
        {"role": "assistant", "content": "6!"},
        {"role": "user", "content": "+4"},
    ]
    tokens_ids, token_str = build_prompt(messages, tok)
    """


