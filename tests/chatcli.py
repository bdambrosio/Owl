import argparse
import json
import socket
import os
import traceback
import ctypes
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.ChoiceResponseValidator import ChoiceResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as lc
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.alphawaveTypes import PromptCompletionOptions

openai_api_key = os.getenv("OPENAI_API_KEY")

temperature=0.1
top_p = 1
max_tokens=100
FORMAT='FORMAT'
prompt_text = 'you are a chatty ai.'
PROMPT = Prompt([
    SystemMessage(prompt_text),
    ConversationHistory('history', .5),
    UserMessage('{{$input}}')
])

#print(f' models: {llm.get_available_models()}')
parser = argparse.ArgumentParser()
#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')

model = ''
modelin = input('model name? ').strip()
if modelin is not None and len(modelin)>1:
    model = modelin.strip()
    models = llm.get_available_models()
    while model not in models:
        print(models)
        modelin = input('model name? ').strip()
        model=modelin

class LLM():
   def __init__(self, memory, osClient, openAIClient, template='alpaca'):
        self.openAIClient=openAIClient
        self.memory = memory
        self.osClient= osClient
        self.template = template # default prompt template.
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()

   def ask(self, input, prompt_msgs, client=None, template=None, temp=0.1, max_tokens=512, top_p=1.0, stop_on_json=False, validator=DefaultResponseValidator()):
      """ Example use:
          class_prefix_prompt = [SystemMessage(f"Return a short camelCase name for a python class supporting the following task. Respond in JSON using format: {{"name": '<pythonClassName>'}}.\nTask:\n{form}")]
          prefix_json = self.llm.ask(self.client, form, class_prefix_prompt, max_tokens=100, temp=0.01, validator=JSONResponseValidator())
          print(f'***** prefix response {prefix_json}')
          if type(prefix_json) is dict and 'name' in prefix_json:
             prefix = prefix_json['name']
      """

      if template is None:
         template = self.template
      if 'gpt' in template:
         client = self.openAIClient
      else:
         client = self.osClient
      options = PromptCompletionOptions(completion_type='chat', model=template,
                                        temperature=temp, top_p= top_p, max_tokens=max_tokens,
                                        stop_on_json=stop_on_json)
      try:
         prompt = Prompt(prompt_msgs)
         #print(f'ask prompt {prompt_msgs}')
         # alphawave will now include 'json' as a stop condition if validator is JSONResponseValidator
         # we should do that for other types as well! - e.g., second ``` for python (but text notes following are useful?)
         response = ut.run_wave (client, {"input":input}, prompt, options,
                                 self.memory, self.functions, self.tokenizer, validator=validator)
         print(f'ask response {response}')
         if type(response) is not dict or 'status' not in response or response['status'] != 'success':
            return None
         content = response['message']['content']
         return content
      except Exception as e:
         traceback.print_exc()
         print(str(e))
         return None
       

def set(text):
    global temperature, top_p, max_tokens
    try:
        if 'temp' in text:
            idx = text.find('=')
            if idx > 0:
                temperature = float(text[idx+1:].strip())
        if 'top_p' in text:
            idx = text.find('=')
            if idx > 0:
                top_p = float(text[idx+1:].strip())
        if 'max_tokens' in text:
            idx = text.find('=')
            if idx > 0:
                max_tokens = int(text[idx+1:].strip())
    except Exception as e:
        print ("parse failed", e)

def setFormat():
    global FORMAT
    if FORMAT:
        format_button.config(text='RAW')
        FORMAT=False
    else:
        format_button.config(text='FORMAT')
        FORMAT=True


def setPrompt(input_text):
    global PROMPT
    input_text = input_text.strip()
    if input_text.startswith("N"):
        input_text = ''
    elif input_text.startswith("H"):
        input_text = 'Respond as a knowledgable and friendly AI, speaking to an articulate, educated, conversant. Limit your response to 100 words where possible. Say "I don\'t know" when you don\'t know."'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("B"):
        input_text = 'Respond as a compassionate, empathetic, self-realized follower of Ramana Maharshi.Limit your response to 100 words where possible.'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("A"):
        input_text = 'Respond as a compassionate, empathetic, counselor familiar with Acceptance Commitment Therapy. Limit your response to 100 words where possible.'
        print(f'prompt set to: {input_text}')
    elif input_text.startswith("F"):
        input_text = 'Respond as a friendly, chatty young woman named Samantha.Limit your response to 100 words where possible.'
        print(f'prompt set to: {input_text}')
    else:
        print(f'valid prompts are N(one), H(elpful), B(hagavan), A(CT), F(riend)')
    
    memory.set('prompt_text', input_text)
    PROMPT = Prompt([
        SystemMessage('{{$prompt_text}}'),
        ConversationHistory('history', .5),
        UserMessage('{{$input}}')
    ])

    
def clear():
    global memory, PREV_POS
    PREV_POS="1.0"
    memory.set('history', [])


functions = FunctionRegistry()
tokenizer = GPT3Tokenizer()
memory = VolatileMemory({'input':'', 'history':[]})
max_tokens = 2000
# Render the prompt for a Text Completion call
def render_text_completion():
    #print(f"\n***** chat memory pre render \n{memory.get('history')}\n")
    as_text = PROMPT.renderAsText(memory, functions, tokenizer, max_tokens)
    #print(as_text)
    text = ''
    if not as_text.tooLong:
        text = as_text.output
    return text

def render_messages_completion():
    #print(f"\n***** chat memory pre render input: \n{memory.get('input')}")
    #print(f"***** chat memory pre render \n{memory.get('history')}\n")
    as_msgs = PROMPT.renderAsMessages(memory, functions, tokenizer, max_tokens)
    msgs = []
    if not as_msgs.tooLong:
        msgs = as_msgs.output
    return msgs


host = '127.0.0.1'
port = 5004

temperature=0.1
top_p = 1.0
max_tokens=256

def run_query(llm, query):
    global model, temperature, top_p, max_tokens, memory
    response = ''
    try:
        msgs = [
            SystemMessage('{{$prompt_text}}'),
            ConversationHistory('history', .5),
            UserMessage('{{$input}}')
        ]
        response = llm.ask(query, msgs, template=model, max_tokens=int(max_tokens), temp=float(temperature), top_p=float(top_p))
        history = memory.get('history')
        history.append({'role':'user', 'content': query.strip()})
        history.append({'role': 'assistant', 'content': response.strip()})
        memory.set('history', history)
    except Exception:
        traceback.print_exc()
    return response

osClient = OSClient(api_key=None)
openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
memory = VolatileMemory({'input':'', 'history':[]})
memory.set('prompt_text', prompt_text)
llm = LLM(memory, osClient, openAIClient, model)

while True:
    ip = input('?')
    if ip.strip().startswith('set '):
        if 'temp' in ip or 'top_p' in ip or 'max_tokens' in ip:
            set(text)
        elif 'format' in ip:
            setFormat()
        elif 'prompt' in ip:
            idx = ip.find('prompt')
            setPrompt(ip[idx+6:].strip())
        elif 'clear' in ip:
            clear()
    else:
        print(run_query(llm, ip.strip()))
