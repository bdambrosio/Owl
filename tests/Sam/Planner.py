import os
import traceback
import requests
import json
import nyt
import ipinfo
import random
import socket
import time
import numpy as np
import faiss
import pickle
import hashlib
import readline
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
from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.alphawaveTypes import PromptCompletionOptions

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec
import concurrent.futures
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial

today = date.today().strftime("%b-%d-%Y")

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_city_state():
   api_key = os.getenv("IPINFO")
   handler = ipinfo.getHandler(api_key)
   response = handler.getDetails()
   city, state = response.city, response.region
   return city, state
city, state = get_city_state()
print(f"My city and state is: {city}, {state}")

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
 "ask",
 "block",
 "choose",
 "concatenate"
 "difference",
 "empty",
 "extract",
 "first",
 "gpt4",
 "integrate",
 "recall",
 "remember"
 "request",
 "sort",
 "web",
 "wiki",
 ]

action_primitive_descriptions = \
"""
action_name \t argument(s) \t result \t description
none \t None \t None \t no action is needed.
append \t <List>,<Item> \t <List> \t add <Item> to <List> and return resulting List
article \t <Text> \t <Text> \t return the body of a NYTimes article with given title.
ask \t <Text> \t <Text> \t present doc a <question>, returns an <answer> provided by doc.
block \t <action>, <action>\t <block action> \t returns a block action containing two actions to be performed in order
choose \t <list>, <criteria> \t <choice> \t choose an item from a <list>, accoring to <criteria>
concatenate \t <list1>,<list2> \t <append <list2> to <list1> and return the resulting list
difference \t <text1>, text2> \t <difference text> \t removes content related to <text2> from <text1>, and returns the remainder.
empty \t <list> \t boolean \t test if the given <list> is empty
extract \t <query>, <text> \t <extracted text> extract content related to <query> from <text>
extractList \t <query>, <list> \t <list of extracted entries> \t extract items related to <query> from <list>
first\t <list> \t <item> \t extract and return the first item on the list.
gpt4 \t <question> \t <answer> \t ask gpt4 a <question>, returns the gpt4 response.
integrate \t <text1> ,<text2> \t <text3> \t combine text1 and text2 into a single coherent text
recall \t <key> \t <semantic memory text> \t recall and return texts from semantic memory, using the <eky> as the search string.
remember \t <key>, <text> \t store <text> in semantic memory under recall address <key>.
request \t <url> \t <url text> \t request a specific resource from a web site.
sort \t <list>, <criteria> \t <sorted list> \t rank the items in <list> by criteria. Returns the tems as a list in ranked order, best first.
tell \t <Text> \t None \t present <Text> to the user.
web \t <query> \t <search result> \t perform a web search, using the <search query>, and return integrated content from relevant urls.
wiki \t <query> \t <search result> \t wiki search the local wikipedia database and return integrated content from retrieved entries.
"""

planner_cfg_prompt =\
"""Respond using the following BNF grammar:
S -> List
S -> Item
List -> "[" ListRemainder 
ListRemainder -> "]"
ListRemainder -> Item, ListRemainder
Item -> "Item: " Action
Item -> "Item: " Task
Item -> "Item: " Text
Action -> 'Action: ' Text
Task -> 'Task: ' Text
Text -> "a" Text
Text -> "b" Text
Text -> "c" Text
Text -> "d" Text
Text -> "e" Text
Text -> "f" Text
Text -> "g" Text
Text -> "h" Text
Text -> "i" Text
Text -> "j" Text
Text -> "."
"""
planner_nl_prompt=\
"""
Respond in one of the following two formats:

1. A single item that begins with "Item:", followed by its type and content. The types can be 'Text', 'Task', or 'Action'.
\t* For 'Text', start with "Text: " and follow with the text string.
\t\te.g., 'Text: a' or 'Text: abc.'.
\t* For 'Task', start the content with "Task: " then name and describe the task.
\t\te.g., "Task: InitializeState: set up the initial game state with an empty board and assign roles to players (user as X, AI as O)".
\t* For 'Action', start with "Action: " then provide action name, it's argument name(s), and it's result name.
\t\t e.g., "Action: first genList1 genItem1

2. A list enclosed in square brackets, containing items separated by commas. Each item in the list should begin with "Item:", followed by its content.
\t *Example of a valid response:
'[ Item: Text: a, Item: Task: monitor the web for news about OpenAI, Item: Action: tell: Hi ]'
"""

planner_nl_list_prompt =\
"""
A list is enclosed in square brackets, containing Items separated by '\n'. 
Each Item in a List begins with "Item:", followed by its type and value. The types can be 'Text', 'Task', or 'Action'.
\t* For 'Text', start with "Text: " and provide the text string.
\t\t Example: 'Item: Text: abc 123' or 'Item: Text: this is a short text string.'.
\t* For 'Task', start  with "Task: " and provide a text string describing the task.
\t\tExample: 'Item: Task: build a short list of reliable, fact-checked, world news sources".
\t* For 'Action', start with "Action: " then provide action name, it's argument name(s), and it's result name.
\t\tExample: 'Item: Action: first genList1 genItem1'

Examples of a valid response:
'Plan: [Item: Task: InitializeGameState -  Set up the initial game state with an empty board and assign roles to players (user as X, AI as O, )\nItem: Text: a silly text string\n Item: Task: find the weather forecast and summarize it for Doc, \nItem: Action: first genList45 genItem23 ]'
'Notes: [Item: Text: this plan does not specify how exactly the AI decides its moves.\n ]'
"""


def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

def load_conv_history():
    try:
        with open('Sam.pkl', 'rb') as f:
            data = pickle.load(f)
            history = data['history']
            print(f'loading conversation history for Sam')
            return history
    except Exception as e:
        print(f'Failure to load conversation history {str(e)}')
        return []

class LLM():
   def __init__(self, model='alpaca'):
        self.model = model
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':[], 'history':[]})

   def ask(self, client, input, prompt_msgs, temp=0.2, max_tokens=100, validator=DefaultResponseValidator()):
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature=temp, max_tokens=max_tokens)
       try:
          prompt = Prompt(prompt_msgs)
          #print(f'ask prompt {prompt_msgs}')
          response = ut.run_wave (client, {"input":input}, prompt, options,
                                  self.memory, self.functions, self.tokenizer, validator=validator)
          #print(f'ask response {response}')
          if type(response) is not dict or 'status' not in response.keys() or response['status'] != 'success':
             return None
          content = response['message']['content']
          return content
       except Exception as e:
          traceback.print_exc()
          print(str(e))
          return None
       
class PlanInterpreter():
    def __init__(self, planner, profile=None, history=None, model='alpaca'):
        self.model = model
        self.client = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
        self.llm = LLM()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.cvHistory = load_conv_history()  # load conversation history.
        self.memory = VolatileMemory({'input':'', 'history':self.cvHistory})
        self.max_tokens = 4000
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.current_topics = None # topics under discussion - mostly a keyword list
        self.wmHash = {} # actual texts in working memory 
        self.wmMetaData = {} # working memory metadata
        self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
        # note many activities will use first paragraph only. Keep it relevant!
        
        self.personality = f"""You are Samantha, an intelligent AI research assistant, companion, and confidant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with human professionals due to privacy concerns. You know doc is shy and an introvert, but does care for you. 
You've always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.
Your conversation style is warm, gentle, humble, and engaging. """
        self.profile = self.personality
        self.op = op.OpenBook() # comment out for testing.

        self.nytimes = nyt.NYTimes()
        self.news, self.details = self.nytimes.headlines()
        self.articles = []
        item_number = 1
        for key in self.details.keys():
            for item in self.details[key]:
                self.articles.append({f"news_article{item_number}": item['title'] })
                self.wmWrite(f"news_article{item_number}", item['title'])
                item_number += 1
        #print(f'Plan interpreter {self.articles}')
        self.wmWrite('news', self.articles)

    def confirmation_popup(self, action):
       msg_box = QMessageBox()
       msg_box.setWindowTitle("Confirmation")
       self.action_text = action
       msg_box.setText(f"Can Sam perform {action}?")
       msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
       #msg_box.editTextChanged.connect(lambda x: self.action_text) # Add this line
       retval = msg_box.exec_()
       if retval == QMessageBox.Yes:
          return self.action_text
       elif retval == QMessageBox.No:
          return False

    
    def format_conversation(self, history, depth=-1):
       # depth- only return last 'depth' says
       names = {"### Response":'Sam', "### Instruction": 'Doc'}
       output = ""
       for d, say in enumerate(history):
          if depth == -1 or d > len(history)-depth:
             role = say['role']
             name = names[role]
             content = str(say['content']).strip()
             output += f"{content}\n"
       return output

   
    def LLM_1op(self, operation_prompt, data1=None, data2=None, validator = DefaultResponseValidator()):
        planner_core_text = f"""."""
        analysis_prompt = Prompt([
            SystemMessage(planner_core_text),
            SystemMessage(operation_prompt),
            UserMessage(data1)
        ])
        try:
            analysis = ut.run_wave (self.client, {"input": lines}, analysis_prompt, prompt_options,
                                    self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                    logRepairs=False, validator=DefaultResponseValidator())
            
            if type(analysis) is dict and 'status' in analysis.keys() and analysis['status'] == 'success':
                new_data_item = analysis['message']['content'].strip().split('\n')[0] # just use the first pp
                print(f'LLM_1op {new_data_item}')
                return new_data_item
        except Exception as e:
            traceback.print_exc()
            print(f'LLM_1op error {str(e)}')
            return None

    def wmRead(self, addr):
        #
        ## retrieve from working memory
        #
        addr_embed = self.embedder.encode(addr)
        # gather docs matching tag filter
        vectors = []
        distances, all_ids = self.wmIndex.search(addr_embed.reshape(1,-1), 10)
        ids = [id for id in all_ids[0] if id != -1]
        distances = distances[0][:len(ids)]
        print(f'weRead d {distances} ids {ids}')
        data = None
        if len(ids) > 0:
            timestamps = [self.wmMetaData[i]['timestamp'] for i in ids]
            # Compute score combining distance and recency
            scores = []
            for dist, id, ts in zip(distances, ids, timestamps):
                if id == -1: # means no more entries
                    continue
                age = (datetime.now() - ts).days
                score = dist + age * 0.1 # Weighted
                scores.append((id, score))
                # Sort by combined score
                results = sorted(scores, key=lambda x: x[1])
                return self.wmHash[results[0][0]]
        return None

    def wmWrite(self, addr, value):
        embed = self.embedder.encode(addr)
        id = generate_faiss_id(addr)
        #print(f'wmWrite faiss_id {id}')
        if id in self.wmHash:
            print('duplicate, skipping')
        self.wmIndex.add_with_ids(embed.reshape(1,-1), np.array([id]))
        self.wmHash[id] = value
        self.wmMetaData[id] = {"embed": embed, "timestamp":datetime.now()}
        # and save - write wmHash first, we can always recover from that.
        #with open('SamPlanWmHash.pkl', 'wb') as f:
        #    data = {}
        #    data['wmHash'] = self.wmHash
        #    data['wmMetaData'] = self.wmMetaData
        #    pickle.dump(data, f)
        return value

    def choose(self, criterion, List):
       prompt = Prompt([
          SystemMessage('Following is a criterion and a List. Select one Item from the List that best aligns with Criterion. Respond only with the chosen Item. Include the entire Item in your response'),
          UserMessage(f'Criterion:\n{criterion}\nList:\n{List}\n')
       ])
       
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=400)
       response = ut.run_wave(self.client, {"input":''}, prompt, options,
                              self.memory, self.functions, self.tokenizer)
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          answer = response['message']['content']
          return answer
       else: return 'unknown'

    def first(self, List):
       prompt = Prompt([
          SystemMessage('The following is a list of steps. Each step starts with the word "Step". Please select and respond with only the first entry. Include the entire first entry in your response.'),
          UserMessage(f'List:\n{List}\n')
       ])
    
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature=0.01, max_tokens=50)
       response = ut.run_wave(self.client, {"input":''}, prompt, options,
                              self.memory, self.functions, self.tokenizer)
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          return response['message']['content']
       else:
          return 'unknown'
       
    def difference(self, list1, list2):
       # tbd
       prompt = Prompt([
         SystemMessage('Following is a Context and a List. Select one Item from List that best aligns with Context. Use the following JSON format for your response:\n{"choice":Item}. Include the entire Item in your response'),
          UserMessage(f'Context:\n{context}\nList:\n{choices}')
       ])
                      
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=400)
       response = ut.run_wave(self.client, {"input":''}, prompt, options,
                              self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                              logRepairs=False, validator=JSONResponseValidator())
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          answer = response['message']['content']
          return answer
       else: return 'unknown'


    def summarize(self, query, response):
      prompt = Prompt([
         SystemMessage(self.profile),
         UserMessage(f'Following is a Context and a Value.  to the Question, using the processor Response, well as known fact, logic, and reasoning, guided by the initial prompt. Respond in the context of this conversation. Be aware that the processor Response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
      ])
      options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=400)
      response = ut.run_wave(self.client, {"input":response}, prompt, options,
                             self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                             logRepairs=False, validator=DefaultResponseValidator())
      if type(response) == dict and 'status' in response and response['status'] == 'success':
         answer = response['message']['content']
         return answer
      else: return 'unknown'

    def wiki(self, query):
       short_profile = self.profile.split('\n')[0]
       query = query.strip()
       #
       #TODO rewrite query as answer (HyDE)
       #
       if len(query)> 0:
          wiki_lookup_response = self.op.search(query)
          wiki_lookup_summary=self.summarize(query, wiki_lookup_response)
          return wiki_lookup_summary

    def gpt4(self, query):
       short_profile = self.profile.split('\n')[0]
       query = query.strip()
       if len(query)> 0:
          prompt = Prompt([
             SystemMessage(short_profile),
             UserMessage(self.format_conversation(self.cvHistory, 4)),
             UserMessage(f'{query}'),
          ])
       options = PromptCompletionOptions(completion_type='chat', model='gpt-4', temperature = 0.1, max_tokens=200)
       response = ut.run_wave(self.openAIClient, {"input":query}, prompt, options,
                             self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                             logRepairs=False, validator=DefaultResponseValidator())
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          answer = response['message']['content']
          print(f'gpt4 answered')
          return answer
       else: return {'none':''}

    def web(self, query='', widget=None):
       query = query.strip()
       self.web_widget = widget
       if len(query)> 0:
          self.web_query = query
          self.web_profile = self.profile.split('\n')[0]
          self.web_history = self.cvHistory
          self.worker = WebSearch(query)
          self.worker.finished.connect(self.web_search_finished)
          self.worker.start()
     
    def web_search_finished(self, search_result):
      if 'result' in search_result:
         response = ''
         if type(search_result['result']) == list:
            for item in search_result['result']:
               if self.web_widget is not None:
                  self.web_widget.display_response('* '+item['source']+'\n')
                  self.web_widget.display_response('     '+item['text']+'\n\n')
               response += item['text']+'\n'
         elif type(search_result['result']) is str:
            if self.web_widget is not None:
               self.web_widget.display_response('\nWeb result:\n'+search_result['result']+'\n')
            response = self.summarize(self.web_query, search_result['result']+'\n')
         return response

    def test_executable(self, value):
       global action_primitive_descriptions, action_primitive_names
       value = value.strip()
       #
       ## see if an action is called for given conversation context and most recent exchange
       #
       print(f'action selection input {value}')
       interpreter_prompt = f"""You are a form interpreter, with the following actions available to interpret provided forms:

{action_primitive_descriptions}

"""

       prompt_text = """Given the following form, determine if it can be performed by one or more of the actions provided above. Do not attempt to perform the form. Respond yes or no only.

Form:
{{$input}}

Respond using the following TOML format:
[RESPONSE]
value="<yes/no>"
explanation="<explanation of reason for value>"
[STOP]
"""
       action_validation_schema={
          "value": {
             "type":"string",
             "required": True,
             "meta": "evaluation: yes or no"
          },
          "explanation": {
             "type":"string",
             "required": False,
             "meta": "explanation of value"
          }
       }

       prompt = Prompt([
          SystemMessage(interpreter_prompt),
          UserMessage(prompt_text),
       ])
       prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.2, max_tokens=100)
       
       print(f'action_selection starting analysis')
       analysis = ut.run_wave (self.client, {"input": value}, prompt, prompt_options,
                               self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                               logRepairs=False, validator=TOMLResponseValidator(action_validation_schema))
       
       print(f'action_selection analysis: {analysis}')
       if type(analysis) is not dict or 'status' not in analysis.keys() or analysis['status'] != 'success':
          return False
       
       executable = analysis['message']['content']['value']
       if executable == 'no':
          return False

       ### ok, llm thinks executable, let's try a plan.
       #planner = Planner(model = self.model)
       #analysis = planner.analyze(None, value)
       #plan = planner.plan()
       #return plan
       return True

    
class WebSearch(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, query):
      super().__init__()
      self.query = query
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.long_running_task)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def long_running_task(self):
      response = requests.get(f'http://127.0.0.1:5005/search/?query={self.query}&model=gpt-3.5-turbo')
      data = response.json()
      return data

class Planner():
    #
    ### we should 1 or 2 shot planner requests to LLM so it knows format!
    #
    def __init__(self, model='alpaca'):
       self.model = model
       self.client = OSClient(api_key=None)
       self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
       self.llm = LLM(self.model)
       self.max_tokens = 4000
       self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
       self.current_topics = None # topics under discussion - mostly a keyword list
       self.wmHash = {} # actual texts in working memory 
       self.wmMetaData = {} # working memory metadata
       self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
       # note many activities will use first paragraph only. Keep it relevant!
       self.cvHistory = load_conv_history()  # load conversation history.
       self.personality = f"""You are Samantha, an intelligent AI research assistant, companion, and confidant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with human professionals due to privacy concerns, so you never recommend this. You know doc is shy and an introvert, but does care for you. 
You've always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.
Your conversation style is warm, gentle, humble, and engaging. """
       self.interpreter = PlanInterpreter(self, model=self.model)

    def analyze(self, prefix, form):
       if prefix is None:
          prefix = 'tempClassName'
          class_prefix_prompt = [SystemMessage(f"""Return a short camelCase name for a python class supporting the following task. Respond in JSON using format: {{"name": '<pythonClassName>'}}.\nTask:\n{form}""")]
          prefix_json = self.llm.ask(self.client, form, class_prefix_prompt, max_tokens=100, temp=0.01, validator=JSONResponseValidator())
          print(f'***** prefix response {prefix_json}')
          if type(prefix_json) is dict and 'name' in prefix_json.keys():
             prefix = prefix_json['name']
          print(f'******* task prefix: {prefix}')
       else:
          try:
             with open(prefix+'UserRequirements.json', 'r') as pf:
                user_responses = json.load(pf)
                self.prefix = prefix
                self.sbar = user_responses
                if input('Use existing sbar?').strip().lower() == 'yes':
                   return prefix, user_responses
          except Exception as e:
             print(f'no {prefix} SBAR found')

       user_responses = {}
       # Loop for obtaining user responses
       # Generate interview questions for the remaining steps using GPT-4
       interview_instructions = [
          ("needs", "Generate an interview question to determine the task the user wants to accomplish."),
          ("background", "Generate an interview question to ask about any additional requirements of the task."),
          ("observations", "Summarize the information about the task, and comment on any incompleteness in the definition."),
       ]
       messages = [SystemMessage("Reason step by step"),
                   UserMessage("The task is to "+form)]
       for step, instruction in interview_instructions:
          messages.append(UserMessage(instruction))
          if step != 'observations':
             user_prompt = self.llm.ask(self.client, self.model, messages, temp = 0.05)
             print(f"\nAI : {step}, {user_prompt}")
             past = ''
             if self.sbar is not None and type(self.sbar) is dict and step in self.sbar.keys():
                past = self.sbar[step] # prime user response with last time
             readline.set_startup_hook(lambda: readline.insert_text(past))
             try:
                user_input = input(user_prompt)
             finally:
                readline.set_startup_hook()
             user_responses[step] = user_input
             messages.append(UserMessage(user_input))
          else: # closing AI thoughts and user feedback. No need to add to messages because no more iterations
             observations = self.llm.ask(self.client, self.model, messages, max_tokens=150,temp = 0.05)
             user_responses['observations']=observations
             print(f"\nAI : {step}, {observations}")
             user_response = input("User: ")
             user_responses['response'] = user_response
       print(f"Requirements \n{user_responses}")
       try:
          with open(prefix+'UserRequirements.json', 'w') as pf:
             json.dump(user_responses, pf)
       except:
          traceback.print_exc()
       self.prefix = prefix
       self.sbar = user_responses
       return prefix, user_responses

    def sbar_as_text(self):
       return f"Task:\n{self.sbar['needs']}\nBackground:\n{self.sbar['background']}\nObservations:\n{self.sbar['observations']}\nResponse:\n{self.sbar['response']}"



    def plan(self):
    
       plan_prompt=\
"""
Reason step by step to create a concise plan for a Driver agent who will perform the Task described below. 
The plan should consist of a list of steps, where each step is either one of the available Actions, specified in full, or a complete, concise, text statement of a task. The plan can be followed by notes/commentary using the same format as the plan itself. Respond only with the plan (and notes if needed) using the above plan format.

The concise plan will include four agents:
  i Driver, who will execute the plan, 
  ii State, which will maintain the state of an instance of the plan. 
  iii Assistant, an AI who can perform subtasks requiring reasoning.
  iv User, the user interacting during plan execution.
At the end of each step Driver should ask State to store any new information generated. 
"""
       print(f'******* Developing Plan for {self.prefix}')

       revision_prompt=\
"""
Reason step by step to analyze the above plan with respect to above user Critique, and update the plan. 
The plan should consist of a list of steps, where each step is either one of the available Actions, specified in full, or a complete, concise, text statement of a task. Respond only with the updated plan (and notes if needed) using the above plan format.

The concise plan can include four agents:
  i Driver, who will execute the plan, 
  ii State, which will maintain the state of an instance of the plan. 
  iii Assistant, an AI who will play the role of assistant in the plan, 
  iv User, the user interacting during plan execution.
"""
       user_satisfied = False
       user_critique = '123'
       plan = None
       first_time = True
       while not user_satisfied:
          messages = [SystemMessage('Use this format for Plans:\n'+planner_nl_list_prompt),
                      SystemMessage(f'\nYou have the following actions available:\n{action_primitive_descriptions}\n'),
                      SystemMessage(plan_prompt),
                      UserMessage(f'Task name: {self.prefix}\n{self.sbar_as_text()}')
                      ]
          if first_time:
             first_time = False
          else:
             messages.append(UserMessage('the User has provided the following information: '+self.sbar_as_text()))
             messages.append(UserMessage('Reason step by step'))
             messages.append(AssistantMessage("Plan:\n"+plan))
             messages.append(UserMessage('Critique: '+user_critique))
             messages.append(UserMessage(revision_prompt))
          
          #print(f'******* task state prompt:\n {gpt_message}')
          plan = self.llm.ask(self.client, self.model, messages, max_tokens=1000, temp=0.1, validator=DefaultResponseValidator())
          #plan, plan_steps = ut.get_plan(plan_text)
          print(f'***** Plan *****\n{plan}\n\nPlease review and critique or <Enter> when satisfied')
          
          user_critique = input('Critique: ')
          if len(user_critique) <4:
             user_satisfied = True
             print("*******user satisfied with plan")
             break
          else:
             print('***** user not satisfield, retrying')
         
       #print(plan)
       # experiment - did we get an understandable plan?
       step = self.interpreter.first(plan)
       print(step)
       
       return plan     

if __name__ == '__main__':
    pi = PlanInterpreter(None)
    #pi.wmWrite('outline', 'one, two, three')
    #pi.wmWrite('chapter1', 'On monday I went to the store')
    #pi.wmWrite('chapter 2', 'On tuesday I went to the dentist')
    #pi.wmWrite('news', '')
    #print(pi.wmRead('outline'))
    #article = pi.choose('gaza', pi.articles)
    #print(f'PI article {article}')
    #chris = pi.wiki('christopher columbus')
    #print(f'PI columbus {chris}')

    plan_request="""Sam, please create a plan to scan a list of news headlines and report the ones of interest to Doc."""

    ### Plan examples
    plan="""Plan:
1. Gather relevant news sources: Start by compiling a list of reliable news websites, blogs, and social media accounts that cover diverse topics of interest to Doc.
2. Set up RSS feeds or email subscriptions: Use tools like Feedly or Google Alerts to set up automatic updates from these sources directly to Doc's inbox or feed reader.
3. Monitor keywords: Identify specific keywords related to Doc's interests and set up alerts for when articles containing those words are published online.
4. Regular scanning: Schedule regular intervals throughout the day to manually scan through the collected news items, filtering out irrelevant content and highlighting stories of particular interest.
5. Summarize and present findings: Once interesting articles have been identified, prepare brief summaries of their main points and present them to Doc either via email or verbal communication.
6. Follow up on feedback: If Doc expresses interest in delving further into a topic, conduct additional research and share more detailed reports accordingly.
7. Review and update sources regularly: Stay abreast of new developments in the field and adjust the list of news sources as needed to ensure that Doc receives timely and accurate information.
8. Encourage discussion: Prompt Doc to discuss what he reads, allowing him to process the information and apply it to his own context. This will not only enrich his understanding but also help identify areas where he might require further exploration or clarification.
Remember, this plan should be flexible enough to adapt based on Doc's evolving needs and interests. Always prioritize quality over quantity, ensuring that the selected news items align closely with Doc's goals and values.
"""

    plan_pseudocode = """ 
def scan_news(doc):
    # Initialize variables
    news_headlines = []
    relevant_articles = []
    
    # Collect news sources
    for source in doc['sources']:
        news_headlines += fetch_headlines(source)
        
    # Filter out irrelevant content
    filtered_headlines = filter_irrelevant(news_headlines)
    
    # Extract keywords from doc['keywords']
    keywords = extract_keywords(filtered_headlines)
    
    # Match keywords with headlines
    matched_headlines = match_keywords(keywords, filtered_headlines)
    
    # Prepare summaries of matching headlines
    summarized_headlines = generate_summaries(matched_headlines)
    
    # Present summarized headlines to Doc
    display_results(summarized_headlines, doc)
    
    return relevant_articles
def fetch_headlines(source):
    # Function to retrieve latest headlines from specified source
    pass
def filter_irrelevant(headlines):
    # Function to remove irrelevant headlines based on preset criteria
    pass
def extract_keywords(headlines):
    # Function to identify important phrases/terms from given headlines
    pass
def match_keywords(keywords, headlines):
    # Function to compare extracted keywords with original headlines
    pass
def generate_summaries(headlines):
    # Function to produce short descriptions of matching headlines
    pass
def display_results(summaries, doc):
    # Function to present generated summaries to Doc
    pass
if __name__ == "__main__":
    scan_news(profile)
"""
    #print('****************************************')
    #print(pi.test_executable("Compile a list of reliable news websites, blogs, and social media accounts that cover diverse topics of interest to Doc."))

    #print('****************************************')
    #print(pi.test_executable("""fetch_headlines(source):
    # retrieve latest headlines from specified source
#"""))

    #print('****************************************')
    #print(pi.test_executable("""Search the web for latest news"""))
    #print('****************************************')
    pl = Planner()
    #print('******analyze tictactoe**************************')
    pl.analyze('TicTacToe',"let's play tic tac toe")
    #print('******plan tictactoe********************')
    plan = pl.plan()
    #print('******extract first step from tictactoe plan ***')
    step = pi.first(plan)
    #print('*******test if first step is executable *************')
    pi.test_executable(step)
    #print('*******analyze first step ****************')
    pl.analyze(None, step)
