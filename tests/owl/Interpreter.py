import os, json, math, time, requests, sys, re
import traceback
import requests
import time
import numpy as np
import faiss
import pickle
from datetime import datetime, date, timedelta
import openai
from promptrix.VolatileMemory import VolatileMemory
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.ChoiceResponseValidator import ChoiceResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient

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
from LLMScript import LLMScript as script

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
    "calc",
    "choose",
    "concatenate"
    "difference",
    "empty",
    "entails",
    "extract",
    "first",
    "for",
    "if",
    "integrate",
    "llm",
    "question",
    "recall",
    "request",
    "rest",
    "return",
    "sort",
    "tell",
    "web",
    "while",
    "wiki",
    ]

action_primitive_descriptions = \
"""This plan language has 6 primitive datatypes: int, str, list, dict, action, and plan. It also includes variables. 
int corresponds to python int type
str corresponds to python str type and must be surrounded by single quote marks (is "'")
list corresponds to python list type
dict corresponds to python dict type
action is a python dict containing the keys 'action', 'arguments', 'result', 'description'
 - 'action' value is one of the 'action' values listed below
 - 'arguments' is either a single item or a tuple of items, where an item is either a literal of one of the six datatypes or a variable. If an item is a variable, its current value is used for execution of the action, as per normal python evaluation.  
 - 'result' is always variable name, whose value is set to the result of the action on execution.
 - 'description' is an optional key containing a textual description of the action
plan is a python dict containing the keys 'name', 'body', and (optionally) 'notes':
 - 'name' is an str
 - 'body' is either: (1) a python list of items of type 'action' or 'plan'; or (2) an str specifying a task for which further planning is needed.
 - 'notes' is a text string containing additional information about the plan
variable is an alphanumeric string prefixed with $ and must be surrounded by single quote marks (ie "'"). It contains a value in one of the six datatypes.

Following is a complete list of all actions, with example 'arguments', 'result', and 'description' values. The example key values are illustrative only:

    {"action": "none", "arguments": "None", "result": "$Trash", "description": "no action is needed."},
    {"action": "append", "arguments": (arg1, arg2), "result": '$resultVar', "description": "append arg1 to arg2, and assign the result to variable $resultVar. arg1 and arg2 must be or hold values of the same type, both either str or list"},
    {"action": "article", "arguments": arg1, "result": '$resultVar', "description": "retrieve the article with title arg1 and assign the article body to $resultVar."},
    {"action": "assign", "arguments": arg1, "result": '$resultVar', "description": "assign the value of arg1 to variable $resultVar"},
    {"action": "calc", "arguments": arg1, "result": '$resultVar', "description": "evaluate arg1 using python, and assign the result to variable $resultVar. This action is for evaluating simple arithmetic expressions. ""},
    {"action": "choose", "arguments": (arg1, arg2), "result": '$resultVar', "description": "choose an item from the list arg1, according to the criteria in arg2, and assign it to variable $resultVar"},
    {"action": "concatenate", "arguments": (arg1, arg2), "result": '$resultVar', "description": "append the list arg2 to the list arg2 and assign the resulting list to variable $resultVar"},
    {"action": "difference", "arguments": (arg1, arg2), "result": '$resultVar', "description": "identify information in arg1 and not in arg2 and assign it to variable $resultVar"},
    {"action": "empty", "arguments": arg1, "result": '$resultVar', "description": "test if arg1 is an empty str or list and assign the boolean True/False accordingly to $resultVar."},
    {"action": "entails", "arguments": arg1, "result": '$resultVar', "description": "test if arg1 information entails (implies, requires-that) the information in arg2 and assign the boolean True/False accordingly to $resultVar."},
    {"action": "extract", "arguments": (arg1, arg2), "result": '$resultVar', "description": "extract content related to arg1 from arg2 and assign it to $resultVar"},
    {"action": "first", "arguments": arg1, "result": '$resultVar', "description": "select the first item in arg1 (an str or list) and assign it to $resultVar."},
    {"action": "for", "arguments": (arg1, arg2, arg3), "result": '$resultVar', "description": "for each element in arg2, assign it as the value of arg1, then run the action or plan arg3."},
    {"action": "if", "arguments": (arg1, arg2), "result": '$resultVar', "description": "Step1: execute the action arg1 and assign the result to $resultVar. Step2: If $resultVar is not False, execute the action arg2. arg1 must be type action, arg2 can be type of action or plan"},
    {"action": "integrate", "arguments": (arg1, arg2), "result": '$resultVar', "description": "combine arg1 and arg2 into a single consolidated text and assign it to variable $resultVar."},
    {"action": "library_research", "arguments": (arg1, arg2), "result": '$resultVar', "description": "search the library for information about the subject in arg1, expanded with the context in arg2), and assign the resulting information to the variable $resultVar."},
    {"action": "llm", "arguments": (arg1, arg2) "result": '$resultVar', "description": "invoke llm with the instruction arg1 and details arg2. Assign the response to variable $resultVar"},
    {"action": "question", "arguments": arg1, "result": '$resultVar', "description": "present arg1 to the user, and assign user response to variable $resultVar."},
    {"action": "recall", "arguments": arg1, "result": '$resultVar', "description": "retrieve arg1 from working memory and assign it to variable $resultVar."},
    {"action": "request", "arguments": arg1, "result": ;$resultVar', "description": "arg1 value must be an str. request a specific web resource with url arg1 and assign the result to variable $resultVar."},
    {"action": "rest", "arguments": arg1, "result": '$resultVar', "description": "arg1 must be a list or an str. return the remainder of the input list after the first element."},
    {"action": "return", "arguments": arg1 "result": '$resultVar', "description": "return from nested plan, assigning value of arg1 in this plan to the value of $resultVar in the name-space of the enclosing plan."},
    {"action": "sort", "arguments": (arg1, arg2), "result": '$resultVar', "description": "arg1 must be a list. rank the items in arg1 by criteria arg2 and assign the sorted list to variable $resultVar. Returns a list in ranked order, best first."},
    {"action": "tell", "arguments": arg1, "result": "$Trash", "description": "present arg1 to the user."},
    {"action": "web", "arguments": arg1, "result": '$resultVar', "description": "perform a web search, using arg1 as the query, and assign the result to variable $resultVar."},
    {"action": "wiki", "arguments": arg1, "result": '$resultVar', "description": "wiki search the local wikipedia database using arg1 as the search string, and assign the result to $resultVar."}


Example Plan (1-shot):

Plan:
{"name": 'Sample Plan,
 "body: 
  [{"action": "assign", "arguments": 'Python-like planning languages', "result": '$topic1'},
   {"action": "assign", "arguments": 'PDDL', "result": '$topic2'},
   {"action": "assign", "arguments": 'GP-PL', result": '$topic3'},
   {"action": "library_research", "arguments": '$topic1', "result": '$overview1'},
   {"action": "library_research", "arguments": '$topic2', "result": '$overview2'},
   {"action": "library_research", "arguments": '$topic3', "result": '$overview3'},
   {"action": "integrate", "arguments": ('$overview1', '$overview2'), "result": '$comparison1'},
   {"action": "integrate", "arguments": "('$comparison1', '$overview3'), "result": '$comparison2'},
   {"action": "assign", "arguments": 'simple plans where dependencies are expressed as conditional control flow', "result": '$focus'},
   {"action": "library_research", "arguments": '$focus', "result": '$focused_research'},
   {"action": "integrate", "arguments": ('$comparison2', '$focused_research'), "result": '$final_report'},
   {"action": "tell", "arguments": '$final_report', "result": '$Trash'}
  ],
 "notes": 'This plan first assigns the three topics of interest to variables. Then, it performs library research on each topic to gather overviews. These overviews are then integrated into a comparison. The plan then assigns the specific focus of the report to a variable and performs focused research on this topic. This focused research is then integrated into the final report, which is presented to the user.'
}
"""

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
        elif dict_item['action'] == 'integrate':
            return self.do_integrate(dict_item)
        elif dict_item['action'] == 'library':
            return self.do_library(dict_item)
        elif dict_item['action'] == 'llm':
            return self.do_llm(dict_item)
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
        elif dict_item['action'] == 'test':
            return self.do_test(dict_item)
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
        args = item['arguments'] # 
        result = item['result']
        if type(result) is not str or not result.startswith('$'):
            self.cot.display_msg(f"result must be a variable name: <{result}")
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

    def substitute(self, item):
        """
        find an argument in working memory. Note str literals will be resolved to vars if possible
        """
        if type(item) is not str:
           return # can only substitute into a string
        positions = [match for match in re.finditer('\\$\\w*', item)]
        positions.reverse() # start from end, otherwise positions will change!
        substituted = item
        for position in positions:
           substituted = substituted[0:position.start()]+str(self.resolve_arg(position[0]))+substituted[position.end():]
        print(f'item {item}\n substituted {substituted}')
        return substituted

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
        action, arg, result = self.parse_as_action(action)
        arg_resolved = self.resolve_arg(arg)
        self.wm.assign(result, arg_resolved)

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
       
        response = self.llm.ask('', prompt, max_tokens=400, temp=0.01)
        if response is not None:
            self.wm.assign(result, response)
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
       
    def do_library(self, action):
        # library_research
        print(f'gpt {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not str:
            raise InvalidAction(f'argument for llm must be a literal or name: {str(arguments)}')       
        prompt_text = self.resolve_arg(arguments)
        substituted_prompt_text = self.substitute(prompt_text)
        prompt = [SystemMessage(prompt_text)]
        response = self.llm.ask("", prompt)
        #self.cot.display_response(response)
        self.wm.assign(result, response)

    def do_llm(self, action):
        # llm takes a single arg, the prompt
        print(f'gpt {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not str:
            raise InvalidAction(f'argument for llm must be a literal or name: {str(arguments)}')       
        prompt_text = self.resolve_arg(arguments)
        substituted_prompt_text = self.substitute(prompt_text)
        prompt = [SystemMessage(prompt_text)]
        response = self.llm.ask("", prompt)
        #self.cot.display_response(response)
        self.wm.assign(result, response)

    def do_request(self, action):
        print(f'request {action}')
        action, arguments, result = self.parse_as_action(action)
        if type(arguments) is not str:
            raise InvalidAction(f'argument for request must be a literal or name: {str(arguments)}')
        title = ' '
        url = self.resolve_arg(arguments)
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
        if type(arguments) is not str:
            raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')       
        value = self.resolve_arg(arguments)
        self.cot.display_response(f'\nTell: {value}\n')

    def test_form(self, form):
        # simple core llm version of test, where argument is a prompt for llm
        # 'is $item1 > 0?'
        # used to eval condition body of control flow forms test, if, while
        print(f'test {form}')
        if type(form) is not str:
            raise InvalidAction(f'argument for test must be a literal or name: {str(form)}')       
        question = self.resolve_arg(form)
        substituted_question = self.substitute(question)
        prompt = [
           SystemMessage("""The user will provide a text to evaluate. 
Read the text, reason about the text and respond 'True' or 'False' according to its truth value.
Respond only with True or False, do not include any introduction or explanation.
"""),
           UserMessage("""Text:

{{$text}}

Answer:""")
        ]
        response = self.llm.ask({"text":substituted_question}, prompt, max_tokens=3, temp=0.01)
        print(f'\nTest: {substituted_question}, result: {response}\n')
        if response is not None:
            response = response.lower()
            if 'yes' in response or 'true' in response:
               return True
            elif 'no' in response or 'false' in response:
               return False
        raise InvalidAction(f'test returned None')
        return False

    def do_test(self, action):
        # simple llm version of test, where argument is a prompt for llm
        # {"action":"testl", "arguments":'is $item1 > 0?', "result":'$item2'}
        print(f'test {action}')
        action, arguments, result = self.parse_as_action(action)
        response = self.test_form(arguments)
        if response is not None:
            if type(response) is str:
               response = response.lower()
            if response or (type(response) is str and ('yes' in response or 'true' in response)):
               self.wm.assign(result, 'True')
               return True
            elif (not response) or (type(response) is str and ('no' in response or 'false' in response)):
               self.wm.assign(result, 'False')
               return False
        raise InvalidAction(f'test_form eval failed')
        return None


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
        action_name = action['action']
        if action_name in ['break', 'continue', 'if', 'return', 'while']:
           return True
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
            # default is execute next instruction, push next inst onto IP stack
            if counter < len(actions)-2: 
                counter += 1
                IP.append((counter, actions))
            if not self.is_controlFlow(action):
               print(f'executing {action}')
               self.do_item(action)
            else:
                # only if, rest tbd
                action, arguments, result = self.parse_as_action(action)
                if type(arguments) is tuple and len(arguments)==2: 
                    test = arguments[0]
                    body = arguments[1]
                else:
                    self.cot.display_response('if needs a tuple: (test, body)\n {arguments}')
                    raise InvalidAction(f'arguments for choose must be a literals or names: {json.dumps(action)}')
                tr = self.test_form(test)
                if tr:
                    do_item(body)
                    self.wm.assign(result, test_result)
                    # now modify IP- push then on stack.
         
          
if __name__ == '__main__':
   import OwlCoT
   cot = OwlCoT.OwlInnerVoice()
   interp = Interpreter(cot)
 
   """
   steps = [{"label": 'one', "action": "assign", "arguments": "Apple", "result": "$item1"},
            {"label": 'two', "action": "tell", "arguments": "$item1", "result": "$Trash"},
            {"label": 'three', "action": "assign", "arguments": 5, "result": "$item2"},
            {"label": 'four', "action": "tell", "arguments": "$item2", "result": "$Trash"},
            {"label": 'five', "action": "assign", "arguments": [], "result": "$item1"},
            {"label": 'six', "action": "tell", "arguments": "$item1", "result": "$Trash"},
            {"label": 'seven', "action": "assign", "arguments": {"key":"value"}, "result": "$item1"},
            {"label": 'eight', "action": "test", "arguments": "is $item1 the same as $item2 ?", "result": "$item3"},
            {"label": 'last', "action": "tell", "arguments": "$item1", "result": "$Trash"},
            {"label": 'last', "action": "tell", "arguments": "$item1", "result": "$Trash"},
            ]
   interp.interpret(steps)
   sys.exit(0)
   steps = [
      {"label": 'one', "action": "request", "arguments": "https://arxiv.org/abs/2311.05584", "result": "$paper_content"},
      {"label": 'two', "action": "llm", "arguments": ("$paper_content", "extract key points"), "result": "$paper_key_points"},
      {"label": 'three', "action": "tell", "arguments": "$paper_key_points", "result":"$Trash"},
      {"label": 'four', "action": "question", "arguments": "Do you want to know more about Q-Learning or other methods?", "result": "$user_choice"},
      {"label": 'five', "action": "web", "arguments": "$user_choice in large language models", "result": "$chosen_method_info"},
      {"label": 'six', "action": "tell", "arguments": "$chosen_method_info", "result":"$Trash"},
      {"label": 'seven', "action": "question", "arguments": "Do you have any other questions?", "result": "$user_question"},
      {"label": 'eight', "action": "llm", "arguments": ("$user_question", "answer"), "result": "$user_question_answer"},
      {"label": 'nine', "action": "tell", "arguments": "$user_question_answer", "result":"$Trash"},
      {"label": 'ten', "action": "none", "arguments": "None", "result": "$Trash"}
   ]

   interp.interpret(steps)
   """
   scriptInterpreter = script(interp, cot)
   scriptInterpreter.fetch('file:///home/bruce/Downloads/owl/tests/owl/arxiv/papers/2305.05181.pdf', 'extract the topic or problem addressed, methods used, data presented, and claims made, and evaluate the significance of this paper', 4000, '$wm123')
   interp.interpret([{"label": 'one', "action": "tell", "arguments": "$wm123", "result":"$Trash"}])
   scriptInterpreter.process(arg='$wm123',
                             instruction='extract key themes and topics of $wm123 in a form useful as a search query.',
                             dest='$wm2',
                             max_tokens=100)
   interp.interpret([{"label": 'one', "action": "tell", "arguments": "$wm2", "result":"$Trash"}])
