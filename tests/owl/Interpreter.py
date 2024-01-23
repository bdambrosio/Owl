import os, json, math, time, requests
import traceback
import requests
import nyt
import ipinfo
import random
import socket
import time
import numpy as np
import faiss
import pickle
import hashlib
#import readline
import nltk
from datetime import datetime, date, timedelta
import openai
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave.MemoryFork import MemoryFork
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.ChoiceResponseValidator import ChoiceResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave_pyexts import utilityV2 as ut
#from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.alphawaveTypes import PromptCompletionOptions

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec
import concurrent.futures
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial
from OwlCoT import ListDialog, LLM, GPT4, TextEditDialog, OPENAI_MODEL3, OPENAI_MODEL4
from workingMemory import WorkingMemory as wm
today = date.today().strftime("%b-%d-%Y")

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']
openai_api_key = os.getenv("OPENAI_API_KEY")

local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour

host = '127.0.0.1'
port = 5004

action_primitive_names = \
   ["none",
    "append",
    "article",
    "assign",
    "choose",
    "concatenate"
    "difference",
    "empty",
    "entails",
    "extract",
    "first",
    "llm",
    "integrate",
    "question",
    "recall",
    "request",
    "sort",
    "tell",
    "web",
    "wiki",
    ]

action_primitive_descriptions = \
   """
[
    {"action": "none", "arguments": "None", "result": "$Trash", "description": "no action is needed."},
    {"action": "append", "arguments": ["$item1", "$item2"], "result": "$item3", "description": "append $item1 to $item2, and assign the resulting list to variable $item3"},
    {"action": "article", "arguments": ["$item1/literal1"], "result": "$item2", "description": "access the article title $item1, use it to retrieve the article body, and assign it to variable $item2."},
    {"action": "assign", "arguments": ["$item1/literal1)"], "result": "$item2", "description": "assign the value of ($item1/literal1) to variable $item2"},
    {"action": "choose", "arguments": ["$item1", "$item2"], "result": "$item3", "description": "choose an item from the list $item1, according to the criteria in $item2, and assign it to variable $item3"},
    {"action": "concatenate", "arguments": ["$item1", "'$item2"], "result": "$item3", "description": "append the list $item2 to the list $item1 and assign the resulting list to variable item3"},
    {"action": "difference", "arguments": ["$item1", "$item2"], "result": "$item3", "description": "identify content in $item1 and not in $item2 and assign it to variable $item3"},
    {"action": "empty", "arguments": ["$item1"], "result": "$item2", "description": "test if $item1 is an empty list and assign the boolean True/False accordingly to $item2."},
    {"action": "entails", "arguments": ["$item1"], "result": "$item2", "description": "test if $item1 content entails (implies, necessarily $item2 is an empty list and assign the boolean True/False accordingly to $item2."},
    {"action": "extract", "arguments": ["$item1/literal1", "$item2"], "result": "$item3", "description": "extract content related to ($item1/literal1) from $item2 and assign it to variable $item3"},
    {"action": "first", "arguments": ["$item1]", "result": "$item2", "description": "select the first item in $item1 and assign it to variable $item2."},
    {"action": "llm", "arguments": ["$item1", "item2"] "result": "$item3", "description": "invoke llm with the instruction $item1 and details $item2. Assign the response to variable $item3"},
    {"action": "integrate", "arguments": ["$item1" ,"$item2"], "result": "$item3", "description": "combine $item1 and $item2 into a single consolidated text and assign it to variable $item3."},
    {"action": "question", "arguments": ["$item1"], "result": "$item2", "description": "access $item1, present it to the user, and assign user response to variable $item2."},
    {"action": "recall", "arguments": ["$item1/literal1"], "result": "$item2", "description": "retrieve $item1 from working memory and assign it to variable $item2."},
    {"action": "request", "arguments": ["$item1"], "result": "$item2", "description": "request a specific web resource with url $item1 and assign the result to variable $item2."},
    {"action": "sort", "arguments": ["$item1", "$item2"], "result": "$item2", "description": "rank the items in $item1 by criteria in $item2 and assign the sorted list to variable $item2. Returns a list in ranked order, best first."},
    {"action": "tell", "arguments": ["$item1"], "result": "$Trash", "description": "present $item1 to the user."},
    {"action": "web", "arguments": ["$item1/literal1"], "result": "$item2", "description": "perform a web search, using ($item1/literal1) as the query, and assign the result to variable $item2."},
    {"action": "wiki", "arguments": ["$item1/literal1"], "result": "$item2", "description": "wiki search the local wikipedia database using ($item1/literal1) as the search string, and assign the result to $item2."}
]

Example Plan (1-shot):

Plan:
[
{"action": "assign", "arguments": ["Apple", "$item1"], "result": "$item2"},
{"action": "assign", "arguments": [], "result": "$item1"},
{"action": "assign", "arguments": ["5"], "result": "$item1"},
{"action": "append", "arguments": ["number1", "$list1"], "result": "$List2"},
{"action": "llm", "arguments": ["What is the meaning of life?", "philosophy"], "result": "$response1"},
{"action": "tell", "arguments": ["$response1"], "result": "$Trash"}
]
"""

def generate_faiss_id(document):
   hash_object = hashlib.sha256()
   hash_object.update(document.encode("utf-8"))
   hash_value = hash_object.hexdigest()
   faiss_id = int(hash_value[:8], 16)
   return faiss_id

def load_conv_history():
   try:
      with open('Owl.pkl', 'rb') as f:
         data = pickle.load(f)
         history = data['history']
         print(f'loading conversation history for Owl')
         return history
   except Exception as e:
      print(f'Failure to load conversation history {str(e)}')
      return []
   
class InvalidAction(Exception):
   # raised by parse_as_action
   pass
       
class Interpreter():
    def __init__(self, owlCoT):
        self.wm = wm()
        self.cot = owlCoT
        self.llm = owlCoT.llm
        self.tokenizer = GPT3Tokenizer()
        self.max_tokens = 7000
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.wmHash = {} # actual texts in working memory 
        self.wmMetaData = {} # working memory metadata
        self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
        # note many activities will use first paragraph only. Keep it relevant!
        
        self.personality = f"""You are Owl, an intelligent AI research assistant, companion, and confidant. We live in {self.cot.city}, {self.cot.state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with human professionals due to privacy concerns.
Your conversation style is warm, gentle, humble, and engaging."""
       
       
    def do_item(self, item):
        dict_item = None
        if type(item) == dict:
            dict_item = item   
        else:
            print(f'do_item item isnt dict, trying loads')
            try:
                dict_item = json.loads(item)
            except:
                print(f'do_item loads failed, trying repair')
                dict_item = self.repair_json(item)
            if type(dict_item) != dict:
                print(f'do_item repair failed {dict_item}')
                return None
        if 'action' not in dict_item:
            self.cot.display_msg(f'item is not an action {item}')
            return 'action not yet implemented'
        elif dict_item['action'] == 'append':
            return self.do_append(dict_item)
        elif dict_item['action'] == 'article':
            return self.do_article(dict_item)
        elif dict_item['action'] == 'assign':
            return self.do_assign(dict_item)
        elif dict_item['action'] == 'choose':
            return self.do_choose(dict_item)
        elif dict_item['action'] == 'concatenate':
            return self.do_concatenate(dict_item)
        elif dict_item['action'] == 'difference':
            return self.do_difference(dict_item)
        elif dict_item['action'] == 'empty':
            return self.do_empty(dict_item)
        elif dict_item['action'] == 'entails':
            return self.do_entails(dict_item)
        elif dict_item['action'] == 'extract':
            return self.do_extract(dict_item)
        elif dict_item['action'] == 'first':
            return self.do_first(dict_item)
        elif dict_item['action'] == 'llm':
            return self.do_llm(dict_item)
        elif dict_item['action'] == 'integrate':
            return self.do_integrate(dict_item)
        elif dict_item['action'] == 'question':
            return self.do_question(dict_item)
        elif dict_item['action'] == 'recall':
            return self.do_recall(dict_item)
        elif dict_item['action'] == 'request':
            return self.do_request(dict_item)
        elif dict_item['action'] == 'sort':
            return self.do_sort(dict_item)
        elif dict_item['action'] == 'tell':
            return self.do_tell(dict_item)
        elif dict_item['action'] == 'web':
            return self.do_web(dict_item)
        elif dict_item['action'] == 'wiki':
            return self.do_wiki(dict_item)
        else:
            self.cot.display_msg(f"action not yet implemented {item['action']}")
        return
   

    def parse_as_action(self, item):
        if type(item) is not dict or 'action' not in item or 'arguments' not in item or 'result' not in item:
            self.cot.display_msg(f'form is not an action/arguments/result {item}')
            raise InvalidAction(str(item))
        args = item['arguments']
        if type(args) is not list:
            args = [args]
        result = item['result']
        if type(result) is not str or not result.startswith('$'):
            self.cot.display_msg(f"result must be a variable name: {result}")
            raise InvalidAction(str(item))
        else:
            return item['action'], args, result

    def resolve_arg(self, item):
        """
        find an argument in working memory. Note str literals will be resolved to vars if possible
        """
        if type(item) is str and item.startswith('$'):
            if self.wm.has(item):
                print(f"resolve_arg returning {self.wm.get(item)['item']}")
                return self.wm.get(item)['item']
            else:
                raise InvalidAction(f"{item} referenced before definition")
        else: # presume a literal
            return item
      
    def do_article(self, titleAddr):
        print(f'article {action}')
        action, arguments, result = self.parse_as_action(action)
        # use OwlCoT to actually retrieve
        #self.wm.assign(result, arguments)
        pass

    def do_assign(self, action):
        #
        ## assign an item or literal as the value of a name
        ## example: {"action":"assign", "arguments":"abc", "result":"pi35"}
        ##   assigns the literal string 'abc' as the value of active memory name pi35 
        ##   if pi35 is not present as a name in active memory, it is created 
        ##   should we recreate key? Not clear, as use case and semantics of key is unclear.
        ##   assume for now assign will be used for simple forms that will be referred to primarily by name, not key.
        print(f'assign {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is list: # assign takes a single argument, extract it from arguments list
            argument0 = arguments[0]
        else:
            argument0 = arguments
        if type(argument0) is not str:
            raise InvalidAction(f'argument for assign must be a literal or name: {json.dumps(action)}')       
        arg0_resolved = self.resolve_arg(argument0)
        self.wm.assign(result, arg0_resolved)

    def do_choose(self, action):
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is list: # 
            arg0 = arguments[0]
            arg1 = arguments[1]
        else:
            self.cot.display_msg('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
        if type(arg0) is not str or type(arg1) is not str:
            raise InvalidAction(f'arguments for choose must be a literals or names: {json.dumps(action)}')       
        criteron = self.resolve_arg(arg0)
        input_list = self.resolve_arg(arg1)
        prompt = Prompt([
            SystemMessage('Following is a criterion and a List. Select one entry from the List that best aligns with Criterion. Respond only with the chosen item. Include the entire item in your response.'),
            UserMessage(f'Criterion:\n{criterion}\nList:\n{input_list}\n')
        ])
       
        options = PromptCompletionOptions(completion_type='chat', model=self.template, temperature = 0.1, max_tokens=400)
        response = self.llm.ask('', prompt, max_tokens=400, temp=0.01)
        if response is not None:
            self.assign(result, response)
        else: 
            raise InvalidAction(f'choose returned None')
                 
    def do_compare(self, action):
        # placeholder
        return self.do_difference(action)

   
    def do_difference(self, action):
        #convert to llm.ask
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is list: # 
            arg0 = arguments[0]
            arg1 = arguments[1]
        else:
            self.cot.display_msg('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
        if type(arg0) is not str or type(arg1) is not str:
            raise InvalidAction(f'arguments for choose must be a literals or names: {json.dumps(action)}')       
        criteron = self.resolve_arg(arg0)
        input_list = self.resolve_arg(arg1)
        # untested
        prompt = Prompt([
            SystemMessage('Following is a Context and a List. Select one Item from List that best aligns with Context. Use the following JSON format for your response:\n{"choice":Item}. Include the entire Item in your response'),
            UserMessage(f'Context:\n{context}\nList:\n{choices}')
        ])
       
        options = PromptCompletionOptions(completion_type='chat', model=self.template, temperature = 0.1, max_tokens=400)
        response = ut.run_wave(self.client, {"input":''}, prompt, options,
                               self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                               logRepairs=False, validator=JSONResponseValidator())
        if type(response) == dict and 'status' in response and response['status'] == 'success':
            answer = response['message']['content']
            self.wm.assign(result, answer)
            return answer
        
        else: return 'unknown'

    def do_extract(self, action):
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is list: # 
            arg0 = arguments[0]
            arg1 = arguments[1]
        else:
            self.cot.display_msg('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
        if type(arg0) is not str or type(arg1) is not str:
            raise InvalidAction(f'arguments for choose must be a literals or names: {json.dumps(action)}')       
        criterion = self.resolve_arg(arg0)
        text = self.resolve_arg(arg1)
        print(f'extract from\n{text}\n')
        prompt = [
            UserMessage(f'Following is a topic and a text. Extract information relevant to topic from the text.Be aware that the text may be partly or completely irrelevant.\nTopic:\n{criterion}\nText:\n{text}'),
            AssistantMessage('')
        ]
        response = self.llm.ask('', prompt, template = self.template, temp=.1, max_tokens=400)
        if response is not None:
            self.cot.create_awm(response, name=result, confirm=False)
            self.cot.display_msg(f'{action}:\n{response}')
            return 
        else: 
            self.cot.create_awm('', name=result, confirm=False)
            self.cot.display_msg(f'{action}:\nNo Text Extracted')
            return 'extract lookup and summary failure'
        
    def do_first(self, action):
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not list and type(arguments) is not dict:
            raise InvalidAction(f'argument for first must be list {arguments}')       
        list = self.resolve_arg(arguments)
        prompt = Prompt([
            SystemMessage('The following is a list. Please select and respond with only the first entry. Include the entire first entry in your response.'),
            UserMessage(f'List:\n{list}\n')
        ])
        
        response = self.llm.ask('', prompt)
        if response is not None:
            self.wm.assign(result, response)
        else:
            return 'unknown'
       
    def do_llm(self, action):
        #
        print(f'gpt {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not list or type(arguments[0]) is not str:
            raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')       
        prompt_text = ""
        for arg in arguments:
            resolved_arg = self.resolve_arg(arg)
            prompt_text += str(resolved_arg)+'\n'
        prompt = [SystemMessage(prompt_text)]
        response = self.llm.ask("", prompt)
        self.cot.display_response(response)
        self.wm.assign(result, response)

    def do_request(self, action):
        print(f'request {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not list or type(arguments[0]) is not str:
            raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')
        arg0 = arguments[0]
        title = ' '
        url = self.resolve_arg(arg0)
        print(f' requesting url from server {url}')
        try:
            print(f"http://127.0.0.1:5005/retrieve?title={title}&url={url}")
            response = requests.get(f"http://127.0.0.1:5005/retrieve/?title={title}&url={url}")
            data = response.json()
        except Exception as e:
            print(f'request failed {str(e)}')
            return {"article": f"\nretrieval failure\n{url}\n{str(e)}"}
        if response is not None:
            self.cot.display_response(f"\nRequest Result:\n {data['result'][:24]}\n")
            self.wm.assign(result, data['result'][:1024])

    def do_tell(self, action):
        #
        print(f'tell {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not list or type(arguments[0]) is not str:
            raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')       
        value = self.resolve_arg(arguments[0])
        self.cot.display_response(value)


    def do_web(self, action):
        #
        print(f'request {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not list or type(arguments[0]) is not str:
            raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')
        arg0 = arguments[0]
        try:
            response = requests.get(f"http://127.0.0.1:5005/search/?query={arg0}")
            data = response.json()
        except Exception as e:
            print(f'request failed {str(e)}')
            return
        if response is not None:
            self.cot.display_response(data)
            self.wm.assign(result, data) # store as a string

    def wiki(self, action):
        # again, where is target var? need to resolve key 'result' value!
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is list: # 
            arg0 = arguments[0]
            arg1 = arguments[1]
        else:
            self.cot.display_response('arguments is not a list\n {arguments}')
            if type(arg0) is not str or type(arg1) is not str:
                raise InvalidAction(f'arguments for choose must be a literals or names: {json.dumps(action)}')
        criteron = self.resolve_arg(arg0)
        input_list = self.resolve_arg(arg1)
        short_profile = profile.split('\n')[0]
        #
        ## use OwlCoT wiki
        return True

    def is_controlFlow(self, action):
        return False
   
    def interpret(self, actions):
        labelled_actions = {}
        for action in actions:
            labelled_actions[action['label']] = action
           
        running = True
        wm = {}
        IP = [(0,actions)]
        #IP = [actions[0]['label']]
        while running and len(IP)>0:
            # pop action label from top of stack
            counter, actions = IP.pop()
            action = actions[counter]
            if not self.is_controlFlow(action):
                if counter < len(actions)-2: # default is execute next instruction
                    counter += 1
                    IP.append((counter, actions))
                    print(f'executing {action}')
                    self.do_item(action)
         
          
if __name__ == '__main__':
   import OwlCoT
   cot = OwlCoT.OwlInnerVoice()
   interp = Interpreter(cot)
 
   steps = [
    {"label": 'one', "action": "request", "arguments": ["https://arxiv.org/abs/2311.05584"], "result": "$paper_content"},
    {"label": 'two', "action": "llm", "arguments": ["$paper_content", "extract key points"], "result": "$paper_key_points"},
    {"label": 'three', "action": "tell", "arguments": ["$paper_key_points"], "result":"$Trash"},
    {"label": 'four', "action": "question", "arguments": ["Do you want to know more about Q-Learning or other methods?"], "result": "$user_choice"},
    {"label": 'five', "action": "web", "arguments": ["$user_choice in large language models"], "result": "$chosen_method_info"},
    {"label": 'six', "action": "tell", "arguments": ["$chosen_method_info"], "result":"$Trash"},
    {"label": 'seven', "action": "question", "arguments": ["Do you have any other questions?"], "result": "$user_question"},
    {"label": 'eight', "action": "llm", "arguments": ["$user_question", "answer"], "result": "$user_question_answer"},
    {"label": 'nine', "action": "tell", "arguments": ["$user_question_answer"], "result":"$Trash"},
    {"label": 'ten', "action": "none", "arguments": ["None"], "result": "$Trash"}
   ]

   interp.interpret(steps)
