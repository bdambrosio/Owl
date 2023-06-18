import openai
import requests
import re
import time
import random
import traceback
import concurrent.futures
import threading as th
import json
import tracemalloc
import os
import linecache
import nltk
import selenium
import inspect
import importlib
from typing import Any, Dict, List, Tuple
import openai
import requests
import time
import traceback
import json
import os
import asyncio
from typing import Any, Dict, List, Tuple
import alphawave_pyexts.LLMClient as client
from promptrix.Prompt import Prompt
from alphawave_agents.PromptCommand import PromptCommand
from promptrix.UserMessage import UserMessage
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave_agents.PromptCommand import CommandSchema, PromptCommandOptions
from alphawave.AlphaWave import AlphaWave
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.MemoryFork import MemoryFork

openai.api_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_KEY")
google_cx = os.getenv("GOOGLE_CX")
GOOGLE = 'google'
USER = 'user'
ASSISTANT = 'assistant'
MODEL = 'wizardLM'
GPT_35 = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4'
FC_T5 = 'fs-t5-3b-v1.0'
VICUNA_7B = 'vicuna-7b'
VICUNA_13B = 'vicuna-13b'
WIZARD_13B = 'wizard-13b'
WIZARD_30B = 'wizard-30b'
GUANACO_33B = 'guanaco-33b'
MODEL = WIZARD_30B
#MODEL = GPT_35
#MODEL = GPT_4
#MODEL = GUANACO_33B

#if not MODEL.startswith('gpt') and MODEL != WIZARD_30B and MODEL != GUANACO_33B:
#  # this for connecting to fastchat server
#  client.initialize(MODEL)

def ask_LLM(model, gpt_message, max_tokens=100, temp=0.7, top_p=1.0, tkroot = None, tkdisplay=None):
    completion = None
    response = ''
    try:
      if not model.lower().startswith('gpt'):
        completion = client.run_query(gpt_message, temp,top_p,max_tokens = max_tokens )
        if completion is not None:
          response = completion

      else:
        stream= openai.ChatCompletion.create(
            model=model, messages=gpt_message, max_tokens=max_tokens, temperature=temp, top_p=1, stop='STOP', stream=True)
        response = ''
        if stream is None:
          return response
        for chunk in stream:
          item = chunk['choices'][0]['delta']
          if 'content' in item.keys():
            if tkdisplay is not None:
              tkdisplay.insert(tk.END, chunk['choices'][0]['delta']['content'])
              if tkroot is not None:
                tkroot.update()
              response += chunk['choices'][0]['delta']['content']
    except:
        traceback.print_exc()
    return response

def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def extract_site(url):
    site = ''
    base= findnth(url, '/',2)
    if base > 2: site = url[:base].split('.')
    if len(site) > 1: site = site[-2]
    site = site.replace('https://','')
    site = site.replace('http://','')
    return site

def extract_domain(url):
    site = ''
    base= findnth(url, '/',2)
    if base > 2: domain = url[:base].split('.')
    if len(domain) > 1: domain = domain[-2]+'.'+domain[-1]
    domain = domain.replace('https://','')
    domain = domain.replace('http://','')
    return domain

def part_of_keyword(word, keywords):
    for keyword in keywords:
        if word in keyword:
            return True
    return False


async def run_wave (client, input, prompt, prompt_options, memory, functions, tokenizer, max_repair_attempts=1, logRepairs=False, validator=DefaultResponseValidator()):
    # Create a wave for the prompt
    fork = MemoryFork(memory)
    for key, value in input.items():
      fork.set(key, value)

    wave = AlphaWave(client=client,
                     prompt=prompt,
                     prompt_options=prompt_options,
                     memory=fork,
                     functions=functions,
                     max_repair_attempts = max_repair_attempts,
                     tokenizer=tokenizer,
                     validator = validator,
                     logRepairs = logRepairs)

    response = await wave.completePrompt()
    # Ensure response succeeded
    return {
      'type': "TaskResponse",
      'status': response['status'],
      'message': response['message']
    }
      

async def get_search_phrase_and_keywords(client, query_string, model, memory, functions, tokenizer):

    prompt = Prompt([UserMessage('Text:\n\n{{$input}}\n\n. Analyze the above Text, using this this JSON template:\n\n{"Phrase": <rewrite of Text as an effective google search phrase>, "Keywords": [keywords in Text],"NamedEntities": [NamedEntities in text]}')])
    response_text=''
    completion = None
    schema = {
      'schema_type':'object',
      'title':'rephrase',
      'description':'rephrase search query',
      'properties':{
        'Phrase': {
          'type': 'string',
          'description': 'search phrase'
        },
        'Keywords': {
          'type': 'array',
          'description': 'list of keywords found'
        },
        'Named-Entities': {
          'type': 'array',
          'description': 'list of named-entities found'
        }
      },
      'required':['Phrase', 'Keywords', 'NamedEntities'],
      'returns':"query answer"
    }

    options = PromptCompletionOptions(completion_type='chat', model=model)
    response = await run_wave(client, {'input':query_string}, prompt, options, memory, functions, tokenizer, validator=JSONResponseValidator(schema))

    if type(response) == dict and 'status' in response and response['status'] == 'success':
        content = response['message']['content']
        phrase = content['Phrase']
        keywords = content['Keywords']
        return phrase, keywords
    else:
        return response, []


def reform(elements):
  #reformulates text extracted from a webpage by unstructured.partition_html into larger keyword-rankable chunks
  texts = [] # a list of text_strings, each of at most *max* chars, separated on '\n' when splitting an element is needed
  paragraphs = []
  total_elem_len = 0
  for element in elements:
    text = str(element)
    total_elem_len += len(text)
    if len(text) < 4: continue
    elif len(text)< 500:
      texts.append(text)
    else:
      subtexts = text.split('\n')
      for subtext in subtexts:
        if len(subtext) < 500:
          texts.append(subtext)
        else:
          texts.extend(nltk.sent_tokenize(subtext))
      
  # now reassemble shorter texts into chunks
  paragraph = ''
  total_pp_len = 0
  for text in texts:
    if len(text) + len(paragraph) < 500 :
      paragraph += ' '+text
    else:
      if len(paragraph) > 0: # start a new paragraph
        paragraphs.append(paragraph)
        paragraph=''
      paragraph += text
  if len(paragraph) > 0:
    paragraphs.append(paragraph+'.\n')
  total_pp_len = 0
  for paragraph in paragraphs:
    total_pp_len += len(paragraph)
  return paragraphs
