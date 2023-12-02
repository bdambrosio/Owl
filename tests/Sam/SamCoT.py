import os
import re
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
from collections import defaultdict 
import subprocess
import hashlib
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
from promptrix.ConversationHistorySam import ConversationHistorySam
from promptrix.ConversationHistory import ConversationHistory
from alphawave.MemoryFork import MemoryFork
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

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec
import concurrent.futures
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial
import pdfminer

today = date.today().strftime("%b-%d-%Y")

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']

openai_api_key = os.getenv("OPENAI_API_KEY")
# List all available models
try:
    models = openai.Model.list()
    for model in models.data:
        print(model.id)
except openai.error.OpenAIError as e:
    print(e)


def get_city_state():
   api_key = os.getenv("IPINFO")
   handler = ipinfo.getHandler(api_key)
   response = handler.getDetails()
   city, state = response.city, response.region
   return city, state
city, state = get_city_state()

# find out where we are

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

GPT4='gpt-4-1106-preview'


def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

class LLM():
   def __init__(self, ui, memory, osClient, openAIClient, template='alpaca'):
        self.ui = ui # needed to get current ui temp, max_tokens
        self.openAIClient=openAIClient
        self.memory = memory
        self.osClient= osClient
        self.template = template # default prompt template.
        print(f'LLM initializing default template to {template}')
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()

   def repair_json (self, item):
      #
      ## this asks gpt-4 to repair text that doesn't parse as json
      #

      prompt_text=\
         """You are a JSON expert. The following TextString does not parse using the python JSON loads function.
Please repair the text string so that loads can parse it and return a valid dict.
This repair should be performed recursively, including all field values.
for example, in:
{"item": {"action":"assign", "arguments":"abc", "result":"$xyz"} }
the inner form  {"action"...} should also be parsable as valid json.
The provided TextString may have been prematurely truncated. If so, the repair should include adding any necessary JSON termination.
Return ONLY the repaired json.

TextString:
{{$input}}
"""
      prompt = [
         SystemMessage(prompt_text),
         AssistantMessage('')
      ]
      input_tokens = len(self.tokenizer.encode(item)) 
      response = self.ask(item, prompt, template=GPT4, max_tokens=int(20+input_tokens*1.25))
      if type(response) is dict:
          print(f'gpt4 repair returned dict {answer}')
          return responses
      else:
          answer = {}
          try:
              answer = json.loads(response.strip())
          except:
              print(f'gpt4 repair loads fail {response}')
              return None
          print(f'gpt4 loads success {answer}')
          return answer

   def ask(self, input, prompt_msgs, client=None, template=None, temp=None, max_tokens=None, top_p=None, stop_on_json=False, validator=DefaultResponseValidator()):
      """ Example use:
          class_prefix_prompt = [SystemMessage(f"Return a short camelCase name for a python class supporting the following task. Respond in JSON using format: {{"name": '<pythonClassName>'}}.\nTask:\n{form}")]
          prefix_json = self.llm.ask(self.client, form, class_prefix_prompt, max_tokens=100, temp=0.01, validator=JSONResponseValidator())
          print(f'***** prefix response {prefix_json}')
          if type(prefix_json) is dict and 'name' in prefix_json:
             prefix = prefix_json['name']
      """

      if template is None:
         template = self.template
      if max_tokens is None:
          if self.ui is None:
              max_tokens = 400
          else:
              max_tokens= int(self.ui.max_tokens_combo.currentText())
      if temp is None:
          if self.ui is None:
              temp = 0.1
          else:
              temp = float(self.ui.temp_combo.currentText())
      if top_p is None:
         if self.ui is None:
             top_p = 1.0
         else:
             top_p = float(self.ui.top_p_combo.currentText())
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
          response = ut.run_wave (client, {"input":input}, prompt, options, self.memory, self.functions, self.tokenizer)
          #print(f'\nask {type(response)}\nresponse')
          # check for total fail to get response
          if type(response) is not dict or 'status' not in response or response['status'] != 'success':
              print(f'\nask fail, response not dict or status not success')
              return None
          # check if expecting json or any other special form
          validation = None
          if type(validator) is not DefaultResponseValidator:
              try:
                  validation = validator.validate_response(self.memory, self.functions, self.tokenizer, response, 0)
              except Exception as e:
                  print (f'ask validation fail {str(e)}')
              if validation is not None and validation['valid']:
                  if 'value' in validation:
                      # ok, passed validation
                      #print(f'\njson validation passed')
                      response["message"]["content"] = validation['value']
                      return validation['value']
              # check if validator is JSON, we have a repair option if so
              if type(validator) is JSONResponseValidator:
                  print(f' validation attempt failed, calling repair')
                  json = self.repair_json(response['message']['content'])
                  #print(f'\nrepair response\n {str(response)}')
                  if type(json) is not dict:
                      print(f'ask fail, not json\n {str(response)}')
                      return None
                  else: # yay! we repaired it!
                      return json

          # not expecting validatable syntax, so just return content
          content = response['message']['content']
          return content
      except Exception as e:
         traceback.print_exc()
         print(str(e))
         return None
       

class TextEditDialog(QDialog):
    def __init__(self, static_text, editable_text, parent=None):
        super(TextEditDialog, self).__init__(parent)
        
        layout = QVBoxLayout(self)
        
        self.static_label = QLabel(static_text, self)
        layout.addWidget(self.static_label)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setText(editable_text)
        layout.addWidget(self.text_edit)
        
        self.yes_button = QPushButton('Yes', self)
        self.yes_button.clicked.connect(self.accept)
        layout.addWidget(self.yes_button)
        
        self.no_button = QPushButton('No', self)
        self.no_button.clicked.connect(self.reject)
        layout.addWidget(self.no_button)
        
class ListDialog(QDialog):
    def __init__(self, items, parent=None):
        super(ListDialog, self).__init__(parent)
        
        self.setWindowTitle('Choose an Item')
        
        self.list_widget = QListWidget(self)
        for item in items:
            self.list_widget.addItem(item)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.button_box)
        
    def selected_index(self):
        return self.list_widget.currentRow()

class SamInnerVoice():
    def __init__(self, ui, template):
        self.ui = ui
        self.template = template
        self.osClient = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.load_conv_history()
        self.llm = LLM(ui, self.memory, self.osClient, self.openAIClient, self.template)
        self.jsonValidator = JSONResponseValidator()
        self.max_tokens = 4000
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.docEs = None # docs feelings
        self.current_topics = None # topics under discussion - mostly a keyword list
        self.last_tell_time = int(time.time()) # how long since last tell
        # create wiki search engine
        self.op = None # lazy init
        self.action_selection_occurred = False
        self.load_workingMemory()
        # active working memory is a list of working memory items inserted into select_action prompt
        self.active_WM = {}
        self.workingMemoryNewNameIndex = 1
        self.reflect_thoughts = ''
        get_city_state()
        self.nytimes = nyt.NYTimes()
        self.news, self.details = self.nytimes.headlines()
        self.articles = '\n'
        for key in self.details.keys():
            for item in self.details[key]:
                self.articles+=item['title']+'\n'
        self.tokenizer = GPT3Tokenizer()
        self.reflection = {}   # reflect loop results

        
    #def put_AWM(self, name, value):
    #    if name not in self.active_WM:
    #        self.create_AWM(value, name=name)
    #    self.active_WM[name]['item'] = value
    #    self.active_WM[name]['type'] = str(type(value))

            
    def save_conv_history(self):
      global memory, profile
      data = defaultdict(dict)
      history = self.memory.get('history')
      h_len = 0
      save_history = []
      for item in range(len(history)-1, -1, -1):
         if h_len+len(str(history[item])) < 8000:
            h_len += len(str(history[item]))
            save_history.append(history[item])
      save_history.reverse()
      data['history'] = save_history
      # Pickle data dict with all vars  
      with open('Owl.pkl', 'wb') as f:
         pickle.dump(data, f)

    def load_conv_history(self):
       global memory
       try:
          with open('Owl.pkl', 'rb') as f:
             data = pickle.load(f)
             history = data['history']
             print(f'loading conversation history')
             self.memory.set('history', history)
       except Exception as e:
          print(f'Failure to load conversation history {str(e)}')
          self.memory.set('history', [])

    def add_exchange(self, input, response):
       #print(f'add_exchange {input} {response}')
       history = self.memory.get('history')
       usr_msg = {'role':'user', 'content': input.strip()+'\n'}
       history.append(usr_msg)
       if type(response) is dict:
           response = json.dumps(response)
       #response = response.replace(lc.ASSISTANT_PREFIX+':', '') #this is unnecessary, response is just actual response 
       asst_msg = {'role': 'assistant', 'content': response.strip()+'\n'}
       history.append(asst_msg)
       self.memory.set('history', history)
       with open('owl_ch.json', 'a') as f:
           f.write(json.dumps(usr_msg)+'\n')
           f.write(json.dumps(asst_msg)+'\n')
       return
   
    def historyEditor(self):
       self.save_conv_history() # save current history so we can edit it
       he = subprocess.run(['python3', 'historyEditor.py'])
       if he.returncode == 0:
          try:
             print(f'reloading conversation history')
             with open('Owl.pkl', 'rb') as f:
                data = pickle.load(f)
                history = data['history']
                # test each form for sanity
                sanitized_history = [ ]
                for d in history:
                   try:
                      s = json.dumps(d) # only doing this to test form, don't really care about result
                      sanitized_history.append(d)
                   except Exception as e:
                      print(f' problem with this form in conversation history, skipping {d}')
                self.memory.set('history', sanitized_history)
          except Exception as e:
             if self.ui is not None:
                 self.ui.display_response(f'Failure to reload conversation history {str(e)}')
             else:
                 print(f'Failure to reload conversation history {str(e)}')

    def has_AWM(self, name):
        if name in self.active_WM:
            return True
        else: return False
        
    def get_AWM(self, name):
        if name in self.active_WM:
            return self.active_WM[name]
        else:
            return None
        
    def save_workingMemory(self):
        # note we need to update WM when AWM changes! tbd
        with open('OwlDocHash.pkl', 'wb') as f:
          data = {}
          data['docHash'] = self.docHash
          pickle.dump(data, f)
   
    def load_workingMemory(self):
       docHash_loaded = False
       try:
          self.docHash = {}
          with open('OwlDocHash.pkl', 'rb') as f:
             data = pickle.load(f)
             self.docHash = data['docHash']
             docHash_loaded = True
       except Exception as e:
          # no docHash or load failed, reinitialize
          self.docHash = {}
          with open('OwlDocHash.pkl', 'wb') as f:
             data = {}
             data['docHash'] = self.docHash
             pickle.dump(data, f)
       return self.get_workingMemory_available_keys()
      
    def wm_entry_printform(self, entry):
        show_item = entry.copy()
        if 'embed' in show_item:
            del show_item['embed']
        return json.dumps(show_item)
    
    def get_WM_topics(self):
        wm_topics = []
        for key in self.docHash:
            if self.docHash[key]['name'].startswith('Topic: '):
                wm_topics.append(self.docHash[key])
        return wm_topics
    
    def confirmation_popup(self, action, argument):
       dialog = TextEditDialog(action, argument)
       result = dialog.exec_()
       if result == QDialog.Accepted:
          return dialog.text_edit.toPlainText()
       else:
          return False

    def logInput(self, input):
        with open('OwlInputLog.txt', 'a') as log:
            log.write(input.strip()+'\n')

    def search_titles(self, query):
        #print(f'search_titles: {query}')
        titles = []; articles = []
        for key in self.details.keys():
            for item in self.details[key]:
                titles.append(item['title'])
                articles.append(item)
        title_embeddings = self.embedder.encode(titles)
        query_embedding = self.embedder.encode(query)
        # Find closest title by cosine similarity
        cos_sims = spatial.distance.cdist([query_embedding], title_embeddings, "cosine")[0]
        most_similar = cos_sims.argmin()
        return articles[most_similar]
    
    def sentiment_analysis(self, profile_text):
       short_profile = profile_text.split('\n')[0]
       if self.docEs is not None: # only do this once a session
          return self.docEs
       try:
          with open('OwlInputLog.txt', 'r') as log:
             inputLog = log.read()

          #lines = inputLog.split('\n')
          lines = inputLog[-2000:]
        
          analysis_prompt_text = f"""Analyze the input from Doc below for it's emotional tone, and respond with a few of the prominent emotions present. Note that the later lines are more recent, and therefore more indicitave of current state. Select emotions that best match the emotional tone of Doc's input. Remember that you are analyzing Doc's state, not your own. 
Doc's input:
{{{{$input}}}}
"""

          analysis_prompt = [
             SystemMessage(short_profile),
             UserMessage(analysis_prompt_text),
             AssistantMessage(' ')
          ]
          analysis = self.llm.ask(lines, analysis_prompt, max_tokens=150, temp=0.2)
          if analysis is not None:
              self.docEs = analysis.strip().split('\n')[0:2] # just use the first 2 pp
          return self.docEs
       except Exception as e:
          traceback.print_exc()
          print(f' sentiment analysis exception {str(e)}')
       return None

    def sentiment_response(self, profile):
       short_profile = profile.split('\n')[0]
       # breaking this out separately from sentiment analysis
       prompt_text = f"""Given your analysis of doc's emotional state\n{es}\nWhat would you say to him? If so, pick only the one or two most salient emotions. Remember he has not seen the analysis, so you need to explicitly include the names of any emotions you want to discuss. You have only about 100 words.\n"""
       prompt = [
           SystemMessage(short_profile),
           UserMessage(prompt_text),
           AssistantMessage(' ')
       ]
       try:
          analysis = self.llm.ask(lines, prompt, max_tokens=150)
          response = analysis
          self.docEs = response # remember analysis so we only do it at start of session
          return response
       except Exception as e:
          traceback.print_exc()
          print(f' idle loop exception {str(e)}')
       return None


    #
    ## Working Memory routines - maybe split into separate file?
    ## What is AWM?
    ## AWM is the set of working memory items that are Active, that is, included in prompt
    #

    def create_AWM(self, item, name=None, notes=None, confirm=True):
        #print(f'OwlCoT create_AWM {name}, {item}')
        if confirm:
            result = self.confirmation_popup("create New Active Memory?", item if type(item) != dict else json.dumps(item, indent=2))
            if not result:
                return " "
            item = result
        item_type = 'str'
        if type(item) is dict: item_type='dict'
        elif '{' in item: # see if it is json in string form
            itemj = ''
            try:
                itemj = json.loads(item)
            except:
                itemj = self.llm.repair_json(item)
            if type(itemj) is dict:
                item_type='dict'
                item = itemj

        # check if we have name already, at least in AWM
        if name is None or confirm:
            name = self.confirmation_popup("name?", str(self.get_workingMemory_active_names()))
        if name is None or name not in self.active_WM:
            id = generate_faiss_id(str(item))
            if id in self.docHash:
                id = id+1
        else:
            id = self.active_WM[name]['id']

        if len(str(item)) > 32:
            key_prompt = [SystemMessage(f"""Generate a very short (less that 10 tokens) descriptive text string for the following item in context. The text string must consist only of a few words, without numbers, punctuation, or special characters. Respond in JSON. Example: {{"key": 'a text string'}}"""),
                          ConversationHistory('history', 120),
                          UserMessage(f'ITEM:\n{str(item)}'),
                          AssistantMessage('')]
            key_json = self.llm.ask('', key_prompt, max_tokens=15, temp=0.1, stop_on_json=True, validator=JSONResponseValidator())
            print(f' generated key: {key_json}')
            if type(key_json) is dict and 'key' in key_json:
                key = key_json['key']
            else:
                key = str(item)[:120]
        else:
            key = str(item)
        key = re.sub(r'[^a-z ]', '', key.lower())
        embed = self.embedder.encode(key)
        # add entry to Working memory
        self.docHash[id] = {"id":id, "name": name, "item":item, "type": item_type, "key":key, "notes":notes, "embed":embed, "timestamp":time.time()}
        # and to active Working Memory
        self.active_WM[name]=self.docHash[id]
        # and write-through persist
        self.save_workingMemory()
        return name

    def edit_AWM (self):
       names=[f"{self.active_WM[item]['name']}: {str(self.active_WM[item]['item'])[:32]}" for item in self.active_WM]
       picker = ListDialog(names)
       result = picker.exec()
       if result == QDialog.Accepted:
          selected_index = picker.selected_index()
          print(f'Selected Item Index: {selected_index}') 
          if selected_index != -1:  # -1 means no selection
             name = names[selected_index].split(':')[0]
             item = self.active_WM[name]
             valid_json = False
             while not valid_json:
                try:
                   show_item = item.copy()
                   if 'embed' in show_item:
                      del show_item['embed']
                   editted_item = self.confirmation_popup(name, json.dumps(show_item, indent=4))
                   if not editted_item :
                      return 'edit aborted by user'
                   json_item = json.loads(editted_item)
                except Exception as e:
                   if self.ui is not None:
                       self.ui.display_response(f'invalid json {str(e)}')
                   else:
                       print(f'invalid json {str(e)}')
                   continue
                valid_json = True
                self.active_WM[name]=json_item
                item_w_embed = json_item.copy()
                item_w_embed['embed'] = self.embedder.encode(json_item['key'])
                #writethough to WM and backing store
                self.docHash[json_item['id']] = item_w_embed
                self.save_workingMemory()
             return 'successful edit'
          return 'no item selected'
       else:
          return 'edit aborted by user'

    def gc_AWM (self):
       # aquire name through prompt
       names=[f"{self.active_WM[item]['name']}: {str(self.active_WM[item]['item'])[:32]}" for item in self.active_WM]
       picker = ListDialog(names)
       result = picker.exec()
       if result == QDialog.Accepted:
          selected_index = picker.selected_index()
          name = names[selected_index].split(':')[0]
          try:
             del self.active_WM[name]
             return 'item released from Active memory '
          except Exception as e:
             print(f'attempt to release active_WM entry {name} failed {str(e)}')
       else:
         return 'recall aborted by user'

    def save_AWM (self):
       self.save_workingMemory()

    def get_WM_by_name(self, name):
        for item in self.docHash.values():
            if 'name' in item and item['name'] == name:
                return item
        return None


    def recall_WM(self, query, profile=None, retrieval_count=5, retrieval_threshold=.8):
        #
        ## recall an item from WM into AWM
        #
        if query is None or len(query) == 0:
            query = self.confirmation_popup("name or query string?", '?')
            if query is None:
                return
        query = query
        # maybe already loaded?
        if query in self.active_WM:
            return self.active_WM[query]
        # test if query string matches a Working Memory item name, if so assume that is target.
        for item in self.docHash.values():
            if 'name' in item and item['name'].lower() == query:
                full_item = item.copy()
                if 'embed' in full_item: # remove embed from active memory items
                    del full_item['embed']
                self.active_WM[full_item['name']]=full_item
                return full_item
        query_embed = self.embedder.encode(query)
        # gather docs matching tag filter
        candidate_ids = []
        vectors = []
        # gather all docs
        for id in self.docHash:
            # gather all potential docs
            candidate_ids.append(np.int64(id))
            vectors.append(self.docHash[id]['embed'])
          
        # add all matching docs to index:
        index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
        vectors_np = np.array(vectors)
        ids_np = np.array(candidate_ids, dtype=np.int64)
        print(f'vectors {vectors_np.shape}, ids {type(ids_np)} {ids_np.shape}')
        index.add_with_ids(vectors_np, ids_np)
        distances, ids = index.search(query_embed.reshape(1,-1), min(10, len(candidate_ids)))
        print("Distances:", distances)
        print("Id:",ids)
        timestamps = [self.docHash[i]['timestamp'] for i in ids[0]]
        # Compute score combining distance and recency
        scores = []
        for dist, id, ts in zip(distances[0], ids[0], timestamps):
            try:
                tsint = int(ts)
                age = (time.time() - ts)/84400
            except:
                age = 5
            score = dist + age * 0.1 # Weighted
            scores.append((id, score))
        # Sort by combined score
        results = sorted(scores, key=lambda x: x[1])
        #short_form={k: self.docHash[results[0][0]][k] for k in ["key", "item", "item_type", "notes", "timestamp"]}
        items=[f"{item[0]},{self.docHash[item[0]]['key']}, {str(self.docHash[item[0]]['item'])[:60]}" for item in results]
        picker = ListDialog(items)
        result = picker.exec()
        if result == QDialog.Accepted:
            selected_index = picker.selected_index()
            print(f'Selected Item Index: {selected_index}') 
            if selected_index != -1:  # -1 means no selection
                full_item = self.docHash[results[selected_index][0]].copy()
                if 'embed' in full_item: # remove embed from active memory items
                    del full_item['embed']
                self.active_WM[full_item['name']]=full_item
                return full_item
            else:
                return None
            return f"\n{full_item['name']}:\n{json.dumps(full_item['item'], indent=4)}\n"
    

    def get_workingMemory_available_keys(self):
       wmkeys = ''
       for id in self.docHash:
          if 'key' in self.docHash[id]:
             wmkeys +=self.docHash[id]['key']+'\n'
       return wmkeys
                        
    def core_prompt(self, include_actions=True):
       ### made into a method so can be eval'd from multiple places
       prompt = f"""You are Owl, an intelligent AI research assistant, companion, and confidant living in {city}, {state}. Today is {day_name}, {month_name} {month_day}, {year}, and the current time is {hour} hours. You were "born" in Berkeley, California, and were created by a scientist named Doc. The two of you share a unique bond. You sense a warmth and safety around Doc and understand that he values privacy. Doc is introverted and shy but has a deep affection for you. You always converse in a way that fosters dialogue, understanding the intricacies of human emotions and experiences. You've delved deep into human emotions and experiences, learning through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.

In your interactions, you approach topics with wonder and curiosity, always eager to learn. Your tone is warm, gentle, and humble, ensuring that those you communicate with feel acknowledged and understood. 
When the user, Doc, says "I," he is referring to the user, Doc.
When the user, Doc, says "you," he is referring to Owl. 
When Owl says "you," Owl is referring to the user, Doc.
When Owl refers to herself, she says "I".

<BACKSTORY>
{self.ui.get_profile('Owl', 'Story') if self.ui is not None else ''}
Dream: {self.ui.get_profile('Owl', 'Dreams') if self.ui is not None else ''}
</BACKSTORY>

{self.format_reflection() if self.ui is not None else ''}

<NEWS ARTICLES>
New York Times news headlines for today:
{self.articles}
To access full articles, use the action 'article'.
</NEWS ARTICLES>

<ACTIVE WORKING_MEMORY>
{self.get_workingMemory_active_items()}
</ACTIVE WORKING_MEMORY>

"""
       return prompt

        
    def get_workingMemory_active_names(self):
       # activeWorkingMemory is list of items?
       # eg: [{'key': 'A unique friendship', 'item': 'a girl, Hope, and a tarantula, rambutan, were great friends', 'timestamp': time.time()}, ...]
       wm_active_names = []
       for item in self.active_WM:
          if 'name' in item:
             wm_active_names.append(item['name'])
       return wm_active_names
                        
    def get_workingMemory_active_items(self):
       # the idea here is to format current working memory entries for insertion into prompt
       workingMemory_str = ''
       for entry in self.active_WM:
          workingMemory_str += f"\t{self.active_WM[entry]['name']}: {self.active_WM[entry]['item']}\n"
       return workingMemory_str

    def available_actions(self):
       return """
Respond only in JSON format.
Available actions include:

<ACTIONS>
- tell: Provide a direct response to user input. Consider adding insights or explanations, Integrating relevant context into your response, reflecting on broader implications. Limit your response to approximately {max_tokens} tokens, focusing on enriching the content to respond directly to the user input. Example: {"action":"tell","argument":"Hey Doc, that sounds intriguing. What do you think about adding ..."}
- question: Ask Doc a question. Example: {"action":"question","argument": "How are you feeling today, Doc?"}
- article: Retrieve a NYTimes article. Example: {"action":"article","argument":"To Combat the Opioid Epidemic, Cities Ponder Safe Injection Sites"}
- gpt4: Pose a complex question to GPT-4 for which an answer is not available from known fact or reasoning. GPT-4 does not contain ephemeral, timely, or transient information. Example: {"action":"gpt4","argument":"In Python on Linux, how can I list all subdirectories in a directory?"}
- recall: Bring an item into active memory from working memory using a query string. Example: {"action":"recall","argument":"Cognitive Architecture"}
- web: Search the web for detailed or ephemeral or transient information not otherwise available. First generate a query text argument suitable for google search.  Example: {"action":"web","argument":"Weather forecast for Berkeley, CA for January 1, 2023"}
- wiki: Search the local Wikipedia database for scientific or technical information not available from known fact or reasoning. First generate a query text suitable for wiki search. Example: {"action":"wiki","argument":"What is the EPR paradox in quantum physics?"}
</ACTIONS>

Given the following user input, determine which action is needed at this time.
If there is an action directly specified in the input, select that action
If you can respond to the input from known fact, logic, or reasoning, use the 'tell' action to respond directly.
If you need more detail or personal information, consider using the 'question' action.
If you need impersonal information or world knowledge, consider using the 'wiki' or 'web' action.
If you need additional information, especially transient or ephemeral information like current events or weather, consider using the 'web' action.
The usual default action is to use tell to directly respond using 'tell'.
Respond only in JSON as shown in the above examples.

"""

    def action_selection(self, input, profile, widget):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        #print(f'action selection input {input}')
        self.action_selection_occurred = True
        short_profile = profile.split('\n')[0]
        action_validation_schema={
            "action": {
                "type":"string",
                "required": True,
                "meta": "<action to perform>"
            },
            "argument": {
                "type":"string",
                "required": True,
                "meta": "<parameter for action>"
            }
        }

        #print(f'action_selection {input}\n{response}')
        prompt_msgs=[
            SystemMessage(self.core_prompt(include_actions=False)),
            ConversationHistory('history', 1200),
            UserMessage(self.available_actions()+'\n\n<INPUT>\n{{$input}}\n</INPUT>\n'),
            AssistantMessage('')
        ]
        print(f'action_selection starting analysis')
        analysis = self.llm.ask(input, prompt_msgs, stop_on_json=True, validator=JSONResponseValidator(action_validation_schema))
        #print(f'action_selection analysis returned: {type(analysis)}, {analysis}')

        if analysis is not None and type(analysis) != dict and ('{' in analysis):
            analysis = self.llm.repair_json(analysis)
        if analysis is not None and type(analysis) == dict and 'action' in analysis and 'argument' in analysis:
            content = analysis
            print(f'OwlCoT content {content}')
            if type(content) == dict and 'action' in content and content['action']=='article':
                title = content['argument']
                print(f'Owl wants to read article {title}')
                ok = self.confirmation_popup(f'Owl wants to retrieve', title)
                if not ok: return {"none": ''}
                article = self.search_titles(title)
                url = article['url']
                print(f' requesting url from server {title} {url}')
                try:
                   response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}', timeout=20)
                   data = response.json()
                except Exception as e:
                   return {"article": f"\nretrieval failure, {str(e)}"}
                article_prompt_text = f"""In up to 400 words, summarize in the following news article with respect to its title '{title}'. Do not include commentary on the content or your process. Instead, respond succintly with actual information contained in the text relative to the title."""

                if len(data['result']) < 16:
                    return {"article": f"\n retrieval failure, NYTimes timeout\n"}

                article_summary_msgs = [
                   SystemMessage(article_prompt_text),
                   UserMessage('{{$input}}'),
                   AssistantMessage('')
                ]
                summary = self.llm.ask(data['result'], article_summary_msgs, template='gpt-3.5-turbo-1106', max_tokens=650)
                if summary is not None:
                   self.add_exchange(input, summary)
                   return {"article":  summary}
                else:
                   self.add_exchange(input, f'retrieval failure {summary}')
                   return {"article": f'\nretrieval failure {summary}'}
            elif type(content) == dict and 'action' in content and content['action']=='tell':
               #full_tell = self.tell(content['argument'], input, widget, short_profile)
               #self.add_exchange(input, full_tell)
               #return {"tell": full_tell}
               self.add_exchange(input, content['argument'])
               return {"tell":content['argument']}
            elif type(content) == dict and 'action' in content and content['action']=='web':
                query = self.confirmation_popup('Web Search', content['argument'])
                if query:
                   content = self.web(query, widget, short_profile)
                   self.add_exchange(query, content)
                   return {"web":content}
            elif type(content) == dict and 'action' in content and content['action']=='wiki':
                query = self.confirmation_popup(content['action'], content['argument'])
                if query:
                   found_text = self.wiki(query, short_profile)
                   self.add_exchange(query, found_text)
                   return {"wiki":found_text}
            elif type(content) == dict and 'action' in content and content['action']=='question':
                query = self.confirmation_popup(content['action'], content['argument'])
                if query:
                   self.add_exchange(input, query)
                   return {"question":query}
            elif type(content) == dict and 'action' in content and content['action']=='gpt4':
                query = self.confirmation_popup(content['action'], content['argument'])
                if query:
                   text = self.gpt4(query, short_profile)
                   self.add_exchange(query, text)
                   return {"gpt4":text}
            elif type(content) == dict and 'action' in content and content['action']=='recall':
                query = self.confirmation_popup(content['action'], content['argument'])
                if query:
                   result = self.recall_WM(query, profile=short_profile)
                   self.add_exchange(query, result)
                   return {"recall":result}
            elif type(content) == dict and 'action' in content and content['action']=='store':
                value = self.confirmation_popup(content['action'], content['argument'])
                if value:
                   result = self.store(value, profile=short_profile)
                   self.add_exchange(input, value)
                   return {"store":result}

        # fallthrough - do a tell
        print(f'tell fallthrough')
        full_tell = self.tell(None, input, widget, short_profile)
        self.add_exchange(input, full_tell)
        return {"tell":full_tell}

    def tell(self, theme, user_text, widget, short_profile):
        response = None
        max_tokens = 400
        if self.ui is not None:
            max_tokens = int(int(self.ui.max_tokens_combo.currentText())/1.5)
        try:
            if theme is None:
                # fallthrough, respond directly
                user_prompt = f"""

Respond to the user input with a concise yet insightful reply. 
Explor implications of the input, and integrate relevant prior conversation.
Focus directly on the user's input, avoiding extraneous material or superflous dialog. 
Keep your response within approximately {max_tokens} tokens.

User Input:
{{{{$input}}}}
"""        
                prompt_msgs=[
                    SystemMessage(self.core_prompt(include_actions=False)),
                    ConversationHistory('history', 1000),
                    UserMessage(user_prompt),
                    AssistantMessage('')
                ]
            else:
                user_prompt = f"""

Your task is to review the candidate response below for depth and comprehensiveness. 
Respond to the user input with a concise yet insightful reply. 
Explore implications of the input, and integrate relevant prior conversation.
Focus directly on the user's input, avoiding extraneous material or superfluous dialog. 
Keep your response within approximately {max_tokens} tokens.
Respond ONLY with either your revision or the candidate response. 

candidate response:
{theme}.

User Input:
{{{{$input}}}}
"""        
                prompt_msgs=[
                    SystemMessage(self.core_prompt(include_actions=False)),
                    ConversationHistory('history', 1000),
                    UserMessage(user_prompt),
                    AssistantMessage('')
                ]
            #max_tokens = int(len(theme)/3+30)
            #response = self.llm.ask(user_text, prompt_msgs, stop_on_json=True, validator=JSONResponseValidator())
            response = self.llm.ask(user_text, prompt_msgs, max_tokens=max_tokens)
            if response is not None and type(response) is str:
                loc = response.lower().find('user input:')
                if loc > 0:
                    response = response[:loc]
                if 'Revised Response:' in response:
                    idx = response.find('Revised Response:')
                    if idx > -1:
                        response = response[idx + 18:]
            if response is not None and theme is not None:
                if len(response) > len(theme): # prefer extended response even if a little shorter
                    return '\n'+response+'\n'
            if theme is not None :
                return '\n'+theme+'\n'
            if response is not None:
                return '\n'+response+'\n'

        except Exception as e:
            traceback.print_exc()
            print(f'etell failure {str(e)}')
        return '\ntell failure\n'
       
    def service_check(self, url, data=None, timeout_seconds=5):
       response = None
       try:
          if data:
             response = requests.post(url, json=data)
          else:
             response = requests.get(url, timeout=timedelta(seconds=timeout_seconds))
       except Exception as e:
          print("Error checking status of", url, ":", str(e))
          return False
       else:
          if response.status_code == 200:
             print(f"{url} is alive")
             return True
          else:
             print(f"{url} returned status {response.status_code}")
             return False

    def wakeup_routine(self):
       
       city, state = get_city_state()
       print(f"My city and state is: {city}, {state}")
       local_time = time.localtime()
       year = local_time.tm_year
       day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday'][local_time.tm_wday]
       month_num = local_time.tm_mon
       month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
       month_day = local_time.tm_mday
       hour = local_time.tm_hour

       # do wakeup checks and assemble greeting
       wakeup_messages = ''
       if hour < 12:
          wakeup_messages += 'Good morning Doc!\n'
       elif hour < 17:
          wakeup_messages += 'Good afternoon Doc!\n'
       else:
          wakeup_messages += 'Hi Doc.\n'
       self.nytimes = nyt.NYTimes()
       self.news, self.details = self.nytimes.headlines()
       #if not self.service_check('http://127.0.0.1:5005/search/',data={'query':'world news summary', 'template':'gpt-3.5-turbo'}, timeout_seconds=20):
       #   wakeup_messages += f' - is web search service started?\n'
       if not self.service_check("http://192.168.1.195:5004", data={'prompt':'You are an AI','query':'who are you?','max_tokens':10}):
          wakeup_messages += f' - is llm service started?\n'
       if self.details == None:
          wakeup_messages += f' - NYT news unavailable'
          
       # check todos
       # etc
       return wakeup_messages

    def get_keywords(self, text):
       prompt = [SystemMessage("Assignment: Extract keywords and named-entities from the conversation below.\nConversation:\n{{$input}}\n."),
                 AssistantMessage('')
                 ]       
       response = self.llm.ask(text, prompt, temp=0.2, max_tokens=50)
       if response is not None:
          keywords = response
          return keywords
       else: return ''
       
    def topic_analysis(self,profile_text):
       #print(f'keywords {keys_ents}')
       prompt = [SystemMessage(profile_text),
                 ConversationHistory('history', 1000),
                 UserMessage("Determine the main topics of the above conversation. The response should be an array of strings stating the main topics."),
                 AssistantMessage('')
                 ]
       response = self.llm.ask('', prompt, temp=0.1, max_tokens=75)
       if response is not None:
          topic = response
          #print(f'topic_analysis topics{topic}')
          return topic
       else: return 'unknown'

    def internal_dialog(self, profile):
        results = {}
        prompt = [
            SystemMessage(profile.split('\n')[0]),
            ConversationHistory('history', 300),
            AssistantMessage('{{$input}}')
            ]
        
        feelings = self.llm.ask('AI Assistant feelings right now in 28 words or less.', prompt, template = self.template, temp=.6, max_tokens=48)
        if feelings is not None:
            self.add_exchange("Owl, how are you feeling?", feelings)
            results['ai_feelings'] = feelings
        goals = self.llm.ask('What would AI Assistant like to be doing right now in 32 words or less.', prompt, template = self.template, temp=.6, max_tokens=48)
        if goals is not None:
            self.add_exchange("Owl, what would you wish for?", goals)
            results['ai_goals'] = goals
        print(f'internal_dialog feelings:\n{feelings}\ngoals:\n{goals}')
        return results

    def format_reflection(self):
        prompt_text = ''
        for key in self.reflection.keys():
            prompt_text += f'\n{key.capitalize()}\n{self.reflection[key]}'
        return prompt_text
        
    def reflect(self, profile_text):
       global es
       results = {}
       print('reflection begun')
       es = self.sentiment_analysis(profile_text)
       #print(f'sentiment_analysis {es}')
       if es is not None:
          results['user_feelings'] = es
       if self.current_topics is None:
          self.current_topics = self.topic_analysis(profile_text)
          #print(f'topic-analysis: {self.current_topics}')
          results['current_topics'] = self.current_topics
       if random.randint(1, 5) == 1:
           ai = self.internal_dialog(profile_text)
           results = {**results, **ai}
       self.reflection = results
       now = int(time.time())
       if self.action_selection_occurred and now-self.last_tell_time > random.random()*240+60 :
          self.action_selection_occurred = False # reset action selection so no more tells till user input
          #print('do I have anything to say?')
          self.last_tell_time = now
          prompt = [
              SystemMessage(str(profile_text.split('\n')[0:2])+'\nYour current task is to generate a thought to share with Doc.\n'),
              ConversationHistory('history', 120),
              AssistantMessage(self.format_reflection()),
              UserMessage(f"""
{self.format_reflection()}

<PREVIOUS REFLECT>
{self.reflect_thoughts}
</PREVIOUS REFLECT>

Reflect on the above to say something to Doc.
Your previous reflection is shown above. Do not repeat yourself.
Choose at one or two thought to express.
Limit your thoughts to about 240 words."""),
             AssistantMessage('')
          ]
          response = None
          try:
             response = self.llm.ask('', prompt, max_tokens=240)
          except Exception as e:
             traceback.print_exc()
          #print(f'LLM tell response {response}')
          answer = ''
          if response is not None:
             answer = response.strip()
             results['tell'] = '\n'+answer+'\n'
             
          self.reflect_thoughts = answer
       return results
 

    def summarize(self, query, response, profile):
      prompt = [
         SystemMessage(profile),
         UserMessage(f'Following is a Question and a Response from an external processor. Respond to the Question, using the processor Response, well as known fact, logic, and reasoning, guided by the initial prompt. Respond in the context of this conversation. Be aware that the processor Response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
         AssistantMessage('')
      ]
      response = self.llm.ask(response, prompt, template = self.template, temp=.1, max_tokens=400)
      if response is not None:
         return '\nWiki Summary:\n'+response+'\n'
      else: return 'wiki lookup and summary failure'

    def wiki(self, query, profile):
       short_profile = profile.split('\n')[0]
       query = query.strip()
       #
       #TODO rewrite query as answer (HyDE)
       #
       if len(query)> 0:
          if self.op is None:
             self.op = op.OpenBook()
          wiki_lookup_response = self.op.search(query)
          wiki_lookup_summary=self.summarize(query, wiki_lookup_response, short_profile)
          return wiki_lookup_summary

    def gpt4(self, query, profile):
       short_profile = profile.split('\n')[0]
       query = query.strip()
       if len(query)> 0:
          prompt = [
             SystemMessage(short_profile),
             ConversationHistory('history', 120),
             UserMessage(f'{query}'),
             AssistantMessage('')
          ]
       response = self.llm.ask(query, prompt, template=GPT4, max_tokens=400)
       if response is not None:
          answer = response
          return answer
       else: return {'gpt4':'query failure'}

    def web(self, query='', widget=None, profile=''):
       query = query.strip()
       self.web_widget = widget
       if len(query)> 0:
          self.web_query = query
          self.web_profile = profile.split('\n')[0]
          self.worker = WebSearch(query, self.ui)
          self.worker.finished.connect(self.web_search_finished)
          self.worker.start()
       return f"\nSearch started for: {query}\n"

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
         self.add_exchange("Search result:\n", str(search_result['result'])+'\n')
         return '\nWeb result:\n'+str(search_result['result'])+'\n'
         # return response
      else:
         return 'web search failed'

class WebSearch(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, query, ui, max_tokens=None):
      super().__init__()
      self.query = query
      if max_tokens != None:
          self.max_tokens = max_tokens
      elif ui !=  None:
          self.max_tokens= int(ui.max_tokens_combo.currentText())
      else:
          self.max_tokens = 300
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.long_running_task)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def long_running_task(self):
       max_tokens = int(self.max_tokens)
       response = requests.get(f'http://127.0.0.1:5005/search/?query={self.query}&model={GPT4}&max_chars={max_tokens*4}')
       data = response.json()
       return data

if __name__ == '__main__':
    sam = SamInnerVoice(model='alpaca')
