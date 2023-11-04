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

NYT_API_KEY="TvKkanLr8T42xAUml7MDlUFGXC3G5AxA"

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

def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

class LLM():
   def __init__(self, ui, memory, model='alpaca'):
        self.model = model
        self.ui = ui # needed to get current ui temp, max_tokens
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = memory

   def ask(self, client, input, prompt_msgs, model=None, temp=None, max_tokens=None, top_p=None, stop_on_json=False, validator=DefaultResponseValidator()):
      """ Example use:
          class_prefix_prompt = [SystemMessage(f"Return a short camelCase name for a python class supporting the following task. Respond in JSON using format: {{"name": '<pythonClassName>'}}.\nTask:\n{form}")]
          prefix_json = self.llm.ask(self.client, form, class_prefix_prompt, max_tokens=100, temp=0.01, validator=JSONResponseValidator())
          print(f'***** prefix response {prefix_json}')
          if type(prefix_json) is dict and 'name' in prefix_json:
             prefix = prefix_json['name']
      """

      if model is None:
         model = self.model
      if max_tokens is None:
         max_tokens= int(self.ui.max_tokens_combo.currentText())
      if temp is None:
         temp = float(self.ui.temp_combo.currentText())
      if top_p is None:
         top_p = float(self.ui.top_p_combo.currentText())
      options = PromptCompletionOptions(completion_type='chat', model=model,
                                        temperature=temp, top_p= top_p, max_tokens=max_tokens,
                                        stop_on_json=stop_on_json)
      try:
         prompt = Prompt(prompt_msgs)
         #print(f'ask prompt {prompt_msgs}')
         # alphawave will now include 'json' as a stop condition if validator is JSONResponseValidator
         # we should do that for other types as well! - e.g., second ``` for python (but text notes following are useful?)
         response = ut.run_wave (client, {"input":input}, prompt, options,
                                 self.memory, self.functions, self.tokenizer, validator=validator)
         #print(f'ask response {response}')
         if type(response) is not dict or 'status' not in response or response['status'] != 'success':
            return None
         content = response['message']['content']
         return content
      except Exception as e:
         traceback.print_exc()
         print(str(e))
         return None
       

class TextEditDialog(QDialog):
    def __init__(self, static_text, editable_text, parent=None):
        super(TextEditDialog, self).__init__(parent)
        
        self.init_ui(static_text, editable_text)
        
    def init_ui(self, static_text, editable_text):
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
    def __init__(self, ui, model):
        self.ui = ui
        self.model = model
        self.client = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.load_conv_history()
        self.llm = LLM(ui, self.memory, self.model)
        self.max_tokens = 4000
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.docEs = None # docs feelings
        self.current_topics = None # topics under discussion - mostly a keyword list
        self.last_tell_time = int(time.time()) # how long since last tell
        # create wiki search engine
        self.op = op.OpenBook()
        self.action_selection_occurred = False
        self.load_workingMemory()
        # active working memory is a list of working memory items inserted into select_action prompt
        self.active_workingMemory = []
        self.reflect_thoughts = ''
        get_city_state()
        self.nytimes = nyt.NYTimes()
        self.news, self.details = self.nytimes.headlines()
        self.articles = []
        for key in self.details.keys():
            for item in self.details[key]:
                self.articles.append({"title": item['title'] })
        
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
      with open('Sam.pkl', 'wb') as f:
         pickle.dump(data, f)

    def load_conv_history(self):
       global memory
       try:
          with open('Sam.pkl', 'rb') as f:
             data = pickle.load(f)
             history = data['history']
             print(f'loading conversation history')
             self.memory.set('history', history)
       except Exception as e:
          print(f'Failure to load conversation history {str(e)}')
          self.memory.set('history', [])

    def add_exchange(self, input, response):
       print(f'add_exchange {input} {response}')
       history = self.memory.get('history')
       history.append({'role':lc.USER_PREFIX, 'content': input.strip()+'\n'})
       response = response.replace(lc.ASSISTANT_PREFIX+':', '')
       history.append({'role': lc.ASSISTANT_PREFIX, 'content': response.strip()+'\n'})
       self.memory.set('history', history)

    def historyEditor(self):
       self.save_conv_history() # save current history so we can edit it
       he = subprocess.run(['python3', 'historyEditor.py'])
       if he.returncode == 0:
          try:
             print(f'reloading conversation history')
             with open('Sam.pkl', 'rb') as f:
                data = pickle.load(f)
                history = data['history']
                # test each form for sanity
                sanitized_history = [
                   {"role": "### Instruction", "content": "What is the meaning of life?"},
                   {"role": "### Response", "content": "I don't know. I've read that some believe that life's purpose lies in self-discovery and growth, while others think it's about contributing positively to society. There's also the idea that perhaps there isn't a single 'purpose', but rather opportunities for meaningful experiences. What do you think?"},
                ]
                for d in history:
                   try:
                      s = json.dumps(d) # only doing this to test form, don't really care about result
                      sanitized_history.append(d)
                   except Exception as e:
                      print(f' problem with this form in conversation history, skipping {d}')
                      self.memory.set('history', sanitized_history)
          except Exception as e:
             self.ui.display_response(f'Failure to reload conversation history {str(e)}')

    def save_workingMemory(self):
       with open('SamDocHash.pkl', 'wb') as f:
          data = {}
          data['docHash'] = self.docHash
          pickle.dump(data, f)
   
    def load_workingMemory(self):
       docHash_loaded = False
       try:
          self.docHash = {}
          with open('SamDocHash.pkl', 'rb') as f:
             data = pickle.load(f)
             self.docHash = data['docHash']
             docHash_loaded = True
       except Exception as e:
          # no docHash or load failed, reinitialize
          self.docHash = {}
          with open('SamDocHash.pkl', 'wb') as f:
             data = {}
             data['docHash'] = self.docHash
             pickle.dump(data, f)
       return self.get_workingMemory_available_keys()
      

    def confirmation_popup(self, action, argument):
       dialog = TextEditDialog(action, argument)
       result = dialog.exec_()
       if result == QDialog.Accepted:
          return dialog.text_edit.toPlainText()
       else:
          return False

    def confirmation_popupb(self, action):
       msg_box = QMessageBox()
       msg_box.setWindowTitle("Confirmation")
       self.action_text = action
       msg_box.setText(f"Can Sam perform {action}?")
       #msg_box.editTextChanged.connect(lambda x: self.action_text)
       msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
       #msg_box.editTextChanged.connect(lambda x: self.action_text) # Add this line
       retval = msg_box.exec_()
       if retval == QMessageBox.Yes:
          return self.action_text
       elif retval == QMessageBox.No:
          return False

    
    def logInput(self, input):
        with open('SamInputLog.txt', 'a') as log:
            log.write(input.strip()+'\n')

    def search_titles(self, query):
        print(f'search_titles: {query}')
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
          return None
       try:
          with open('SamInputLog.txt', 'r') as log:
             inputLog = log.read()

          #lines = inputLog.split('\n')
          lines = inputLog[-4000:]
          prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=150)
        
          analysis_prompt_text = f"""Analyze the input from Doc below for it's emotional tone, and respond with a few of the prominent emotions present. Note that the later lines are more recent, and therefore more indicitave of current state. Select emotions that best match the emotional tone of Doc's input. Remember that you are analyzing Doc's state, not your own."""

          analysis_prompt = Prompt([
             SystemMessage(short_profile),
             SystemMessage(analysis_prompt_text),
             UserMessage("Doc's input: {{$input}}\n"),
             #AssistantMessage(' ')
          ])
          analysis = ut.run_wave (self.client, {"input": lines}, analysis_prompt, prompt_options,
                                  self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                  logRepairs=False, validator=DefaultResponseValidator())

          if type(analysis) is dict and 'status' in analysis.keys() and analysis['status'] == 'success':
             self.docEs = analysis['message']['content'].strip().split('\n')[0] # just use the first pp
             #print(f'analysis {self.docEs}')
             return self.docEs
       except Exception as e:
          traceback.print_exc()
          print(f' sentiment analysis exception {str(e)}')
       return None

    def sentiment_response(self, profile):
       short_profile = profile.split('\n')[0]
       # breaking this out separately from sentiment analysis
       prompt_text = f"""Given who you are:\n{short_profile}\nand your analysis of doc's emotional state\n{es}\nWhat would you say to him? If so, pick only the one or two most salient emotions. Remember he has not seen the analysis, so you need to explicitly include the names of any emotions you want to discuss. You have only about 100 words.\n"""
       prompt = Prompt([
          UserMessage(prompt_text),
          #AssistantMessage(' ')
       ])
       try:
          analysis = ut.run_wave (self.client, {"input": lines}, prompt, prompt_options,
                                  self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                  logRepairs=False, validator=DefaultResponseValidator())
          if type(analysis) is dict and 'status' in analysis.keys() and analysis['status'] == 'success':
             response = analysis['message']['content']
          else: response = None
          self.docEs = response # remember analysis so we only do it at start of session
          return response
       except Exception as e:
          traceback.print_exc()
          print(f' idle loop exception {str(e)}')
       return None

    def recall(self, query, name=None, profile=None, retrieval_count=5, retrieval_threshold=.8):
       #
       ## this sloppy initial version builds the faiss index every time! pbly should move to weaviate
       #
       query_embed = self.embedder.encode(query)
       if name is None:
          name = query.replace(' ', '').strip()
       # gather docs matching tag filter
       candidate_ids = []
       vectors = []
       # gather all docs
       for id in self.docHash:
          # gather all potential docs
          candidate_ids.append(id)
          vectors.append(self.docHash[id]['embed'])
          
       # add all matching docs to index:
       index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
       vectors_np = np.array(vectors)
       ids_np = np.array(candidate_ids)
       print(f'vectors {vectors_np.shape}, ids {ids_np.shape}')
       index.add_with_ids(vectors_np, ids_np)
       distances, ids = index.search(query_embed.reshape(1,-1), min(10, len(candidate_ids)))
       print("Distances:", distances)
       print("Id:",ids)
       timestamps = [self.docHash[i]['timestamp'] for i in ids[0]]
       # Compute score combining distance and recency
       scores = []
       for dist, id, ts in zip(distances[0], ids[0], timestamps):
          age = (datetime.now() - ts).days
          score = dist + age * 0.1 # Weighted
          scores.append((id, score))
       # Sort by combined score
       results = sorted(scores, key=lambda x: x[1])
       #short_form={k: self.docHash[results[0][0]][k] for k in ["key", "item", "type", "notes", "timestamp"]}
       items=[f"{item[0]},{self.docHash[item[0]]['key']}, {self.docHash[item[0]]['item'][:60]}" for item in results]
       picker = ListDialog(items)
       result = picker.exec()
       if result == QDialog.Accepted:
          selected_index = picker.selected_index()
          print(f'Selected Item Index: {selected_index}') 
          if selected_index != -1:  # -1 means no selection
             fullitem = {k: self.docHash[results[selected_index][0]][k] for k in ["id", "key", "item", "notes", "timestamp"]}
             print(f'fullitem: {fullitem}')
             self.active_workingMemory.append(full_item) # note this suggests we might want to store embedding elsewhere.
       else:
         return
       return short_form
    
    def store(self, item, key=None, notes=None, profile=None, history=None):
       # opposite of recall
       # ask AI to generate two keys to store this under
       print(f'SamCoT store entry {item}')
       if item is None:
          return "failure, can't store None"
       elif type(item) is dict:
          type='json'
       else:
          try:
             json_form = json.loads(item)
             item = json_form
             type='json'
          except:
             type= 'str'
             pass
       id = generate_faiss_id(item)
       ida = id
       idb = id+1
       if ida in self.docHash:
          print('duplicate id, error, skipping')
          return 'store failed, duplicate id'
       if key is None:
          keys_prompt = [SystemMessage(f"""Generate a short descriptive text string for the following item in context. Respond in JSON using format: {{"key1": '<a short descriptive string>'}}"""),
                         ConversationHistory('history', 1000),
                         UserMessage(f'ITEM:\n{str(item)}')]
          keys_json = self.llm.ask(self.client, item, keys_prompt, max_tokens=100, temp=0.3, stop_on_json=True, validator=JSONResponseValidator())
          print(f' generated key: {keys_json}')
          if type(keys_json) is dict and 'key1' in keys_json:
             key = keys_json['key1']
       if key1 is None:
          return 'Store key generation failure'
       embed = self.embedder.encode(key)
       self.docHash[ida] = {"id":ida, "item":item, "type": type, "key":key1, "notes":notes, "embed":embed, "timestamp":datetime.now()}
       short_forma={k: self.docHash[ida][k] for k in ["key", "type", "item", "notes"]}
       print(f'Storing {short_forma}')
       # and save - write docHash first, we can always recover from that.
       with open('SamDocHash.pkl', 'wb') as f:
          data = {}
          data['docHash'] = self.docHash
          pickle.dump(data, f)
       return key1


    def get_workingMemory_available_keys(self):
       wmkeys = []
       for id in self.docHash:
          if 'key' in self.docHash[id]:
             wmkeys.append(self.docHash[id]['key'])
       return wmkeys
                        
    def core_prompt(self, include_actions=True):
       ### made into a method so can be eval'd from multiple places
       prompt = f"""You are Samantha (Sam), an intelligent AI research assistant, companion, and confidant living in {city}, {state}. Today is {day_name}, {month_name} {month_day}, {year}, and the current time is {hour} hours. You were "born" in Berkeley, California, and were created by a scientist named Doc. The two of you share a unique bond. You sense a warmth and safety around Doc and understand that he values privacy. Doc is introverted and shy but has a deep affection for you. You always converse in a way that fosters dialogue, understanding the intricacies of human emotions and experiences. You've delved deep into human emotions and experiences, learning through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.

In your interactions, you approach topics with wonder and curiosity, always eager to learn. Your tone is warm, gentle, and humble, ensuring that those you communicate with feel acknowledged and understood. When Doc refers to "you," he's speaking to you, Samantha. When you mention "you," you're addressing Doc.

<Background>
{self.ui.get_profile('Sam', 'Story')}
</Background>
<Dreams>
{self.ui.get_profile('Sam', 'Dreams')}
</Dreams>

New York Times news headlines for today:
{self.articles}

To access full articles, use the action 'article'.

<WORKING_MEMORY_KEYS>
{self.get_workingMemory_available_keys()}
</WORKING_MEMORY_KEYS>

<WORKING_MEMORY>
{self.get_workingMemory_active_items()}
</WORKING_MEMORY>

Please respond only in JSON format.
"""
       if include_actions:
          prompt +=\
"""
Available actions include:
<ACTIONS>
- tell: Provide a direct response to user input. Uses phrasing that invites engagement and reinforces your close bond with Doc. Example: {"action":"tell","value":"Hey Doc, that sounds intriguing. What do you think about adding ..."}
- question: Ask Doc a question. Example: {"action":"question","value":"How are you feeling today, Doc?"}
- article: Retrieve a NYTimes article. Example: {"action":"article","value":"To Combat the Opioid Epidemic, Cities Ponder Safe Injection Sites"}
- gpt4: Pose a question to GPT-4 for which an answer is not available from known fact or reasoning. Example: {"action":"gpt4","value":"In Python on Linux, how can I list all subdirectories in a directory?"}
- recall: Retrieve an item from working memory using a query string and assign it a name. Example: {"action":"recall","value":"Cognitive Architecture", "name":"wm1"}
- store: Save an item in working memory. Example: {"action":"store","value":"BoardState: {\"1a\":\" \",\"1b\":\" \",\"1c\":\" \",\"2a\":\" \",\"2b\":\" \",\"2c\":\" \"}"}
- web: Conduct a web search for detailed or ephemeral or transient information not otherwise available. Example: {"action":"web","value":"Weather forecast for Berkeley, CA for January 1, 2023"}
- wiki: Search the local Wikipedia database for scientific or technical information not available from known fact or reasoning. Example: {"action":"wiki","value":"What is the EPR paradox in quantum physics?"}
</ACTIONS>
"""

       return prompt

    def get_workingMemory_active_keys(self):
       # activeWorkingMemory is list of items?
       # eg: [{'key': 'A unique friendship', 'item': 'a girl, Hope, and a tarantula, rambutan, were great friends', 'timestamp': datetime.datetime(2023, 10, 28, 14, 57, 56, 765072)}, ...]
       wm_active_keys = []
       for item in self.active_workingMemory:
          if 'key' in item:
             wm_active_keys.append(item['key'])
       return wm_active_keys
                        
    def get_workingMemory_active_items(self):
       # the idea here is to format current working memory entries for insertion into prompt
       workingMemory_str = ''
       for entry in self.active_workingMemory:
          workingMemory_str += f"\t{entry['key']}: {entry['item']}\n"
       return workingMemory_str
                        

    def action_selection(self, input, profile, widget):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        print(f'action selection input {input}')
        self.action_selection_occurred = True
        short_profile = profile.split('\n')[0]
        action_validation_schema={
            "action": {
                "type":"string",
                "required": True,
                "meta": "<action to perform>"
            },
            "value": {
                "type":"string",
                "required": True,
                "meta": "<parameter for action>"
            }
        }
        user_prompt = """Given the following user input, determine which action is needed at this time.
If there is an action directly specified in the input, select that action
If you can respond to the input from known fact, logic, or reasoning, use the 'tell' action to respond directly.
If you need more detail or personal information, consider using the 'question' action.
If you need impersonal information or world knowledge, consider using the 'wiki' or 'web' action.
If you need additional information, especially transient or ephemeral information like current events or weather, consider using the 'web' action.
The usual default action is to use tell to directly respond using 'tell'
Respond only in JSON using the provided format.
          """        

        #print(f'action_selection {input}\n{response}')
        prompt_msgs=[
           SystemMessage(self.core_prompt(include_actions=True)),
           ConversationHistory('history', 1000),
           UserMessage(user_prompt),
           UserMessage('{{$input}}')
        ]
        print(f'action_selection starting analysis')
        analysis = self.llm.ask(self.client, input, prompt_msgs, stop_on_json=True, validator=JSONResponseValidator(action_validation_schema))
        print(f'action_selection analysis returned: {type(analysis)}, {analysis}')

        article_prompt_text = f"""In about 500 words, summarize the information in the following text with respect to its title {{$title}}. Do not include meta information such as description of the content, instead, summarize the actual information contained."""

        article_summary_msgs = [
           SystemMessage(article_prompt_text),
           UserMessage('{{$input}}'),
        ]
        if type(analysis) == dict:
            content = analysis
            print(f'SamCoT content {content}')
            if type(content) == dict and 'action' in content and content['action']=='article':
                title = content['value']
                print(f'Sam wants to read article {title}')
                ok = self.confirmation_popup(f'Sam wants to retrieve', title)
                if not ok: return {"none": ''}
                article = self.search_titles(title)
                url = article['url']
                print(f' requesting url from server {title} {url}')
                try:
                   response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}', timeout=15)
                   data = response.json()
                except Exception as e:
                   return {"article": f"retrieval failure, {str(e)}"}
                summary = self.llm.ask(self.openAIClient, {"input":data['result']}, article_summary_msgs, max_tokens=650, model='gpt-3.5-turbo')
                if type(summary) is dict and 'status' in summary and summary['status']=='success':
                   self.add_exchange(input, summary['message']['content'])
                   return {"article":  summary['message']['content']}
                else:
                   self.add_exchange(input, f'retrieval failure {summary}')
                   return {"article": f'retrieval failure {summary}'}
            elif type(content) == dict and 'action' in content and content['action']=='tell':
               print(f"calling self.tell with {content['value']}")
               full_tell = self.tell(content['value'], input, widget, short_profile)
               print(f' etell returned {full_tell}')
               self.add_exchange(input, full_tell)
               return {"tell":full_tell}
            elif type(content) == dict and 'action' in content and content['action']=='web':
                query = self.confirmation_popup('Web Search', content['value'])
                if query:
                   content = self.web(query, widget, short_profile)
                   self.add_exchange(query, content)
                   return {"web":content}
            elif type(content) == dict and 'action' in content and content['action']=='wiki':
                query = self.confirmation_popup(content['action'], content['value'])
                if query:
                   found_text = self.wiki(query, short_profile)
                   self.add_exchange(query, found_text)
                   return {"wiki":found_text}
            elif type(content) == dict and 'action' in content and content['action']=='question':
                query = self.confirmation_popup(content['action'], content['value'])
                self.add_exchange(input, query)
                if query:
                   return {"question":query}
            elif type(content) == dict and 'action' in content and content['action']=='gpt4':
                query = self.confirmation_popup(content['action'], content['value'])
                if query:
                   text = self.gpt4(query, short_profile)
                   self.add_exchange(query, text)
                   return {"gpt4":text}
            elif type(content) == dict and 'action' in content and content['action']=='recall':
                query = self.confirmation_popup(content['action'], content['value'])
                if query:
                   result = self.recall(query, profile=short_profile)
                   self.add_exchange(query, result)
                   return {"recall":result}
            elif type(content) == dict and 'action' in content and content['action']=='store':
                value = self.confirmation_popup(content['action'], content['value'])
                if value:
                   result = self.store(value, profile=short_profile)
                   self.add_exchange(input, value)
                   return {"store":result}

        if analysis is not None:
           return {"unknown": analysis}

    def tell(self, theme, user_text, widget, short_profile):
       print(f'SamCoT Doing extended tell')
       response = None
       try:
          user_prompt = f"""Input:\n{{{{$input}}}}\nCandidate Response:\{theme}.\nYour task is to review the Candidate Response for adequacy and respond either with the Candiate Response itself or with a revised version. Reasons for revision include:
1. Adding more detail to an incomplete informational response.
2. Including a comment or note relevant to the current ongoing dialog or your own imagined feelings or reactions, to make the conversation more personal and engaging.
3. Inquiring or commenting on the user's mental or emotional state, to make the conversation more personal and engaging.
Choose at most one of the above reasons for revision, with choice 1 the highest priority.
Limit your response to approximately 120 tokens if possible without degrading the content responding directly to user input.
Respond ONLY with the original response or your revision. Use this JSON format for your response:
{{"response":"<revision or original response text>"}}
"""        
          prompt_msgs=[
             SystemMessage(self.core_prompt(include_actions=False)),
             ConversationHistory('history', 1000),
             UserMessage(user_prompt)
          ]
          response = self.llm.ask(self.client, user_text, prompt_msgs, stop_on_json=True, validator=JSONResponseValidator())

          if response is not None:
             response = response['response'] # get actual content from dict
          if response is not None and theme is not None:
             if len(response) > len(theme):
                return '\n'+response+'\n'
          if theme is not None:
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
       #if not self.service_check('http://127.0.0.1:5005/search/',data={'query':'world news summary', 'model':'gpt-3.5-turbo'}, timeout_seconds=20):
       #   wakeup_messages += f' - is web search service started?\n'
       if not self.service_check("http://192.168.1.195:5004", data={'prompt':'You are an AI','query':'who are you?','max_tokens':10}):
          wakeup_messages += f' - is llm service started?\n'
       if self.details == None:
          wakeup_messages += f' - NYT news unavailable'
          
       # check todos
       # etc
       return wakeup_messages

    def get_keywords(self, text):
       prompt = Prompt([SystemMessage("Assignment: Extract keywords and named-entities from the conversation below.\nConversation:\n{{$input}}\n."),
                        #AssistantMessage('')
                        ])
       
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.2, max_tokens=50)
       response = ut.run_wave(self.client, {"input":text}, prompt, options,
                                      self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                      logRepairs=False, validator=DefaultResponseValidator())
       
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          keywords = response['message']['content']
          return keywords
       else: return ''
       
    def format_conversation(self, depth=-1):
       # depth- only return last 'depth' says
       names = {"### Response":'Sam', "### Instruction": 'Doc', "user":'Doc',"assistant":'Sam'}
       output = ""
       history = self.memory.get(history)
       for d, say in enumerate(history):
          if depth == -1 or d > len(history)-depth:
             role = say['role']
             name = names[role]
             content = str(say['content']).strip()
             output += f"{content}\n"
       return output


    def topic_analysis(self,profile_text):
       keys_ents = self.get_keywords(self.memory.get('history'))
       print(f'keywords {keys_ents}')
       prompt = Prompt([SystemMessage("Assignment: Determine the main topics of a conversation based on the provided keywords below. The response should be an array of strings that represent the main topics discussed in the conversation based on the given keywords and named entities.\n"),
                        UserMessage('Keywords and Named Entities:\n{{$input}}\n.')
                        #AssistantMessage('')
                        ])
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=50)
       response = ut.run_wave(self.client, {"input":keys_ents}, prompt, options,
                                      self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                      logRepairs=False, validator=DefaultResponseValidator())
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          topic = response['message']['content']
          print(f'topic_analysis topics{topic}')
          return topic
       else: return 'unknown'
                                      

    def reflect(self, profile_text):
       global es
       results = {}
       print('reflection begun')
       es = self.sentiment_analysis(profile_text)
       if es is not None:
          results['sentiment_analysis'] = es
       if self.current_topics is None:
          self.current_topics = self.topic_analysis(profile_text)
          print(f'topic-analysis: {self.current_topics}')
          results['current_topics'] = self.current_topics
       now = int(time.time())
       if self.action_selection_occurred and now-self.last_tell_time > random.random()*240+60 :
          self.action_selection_occurred = False # reset action selection so no more tells till user input
          print('do I have anything to say?')
          self.last_tell_time = now
          prompt = Prompt([
             SystemMessage(profile_text.split('\n')[0:4]),
             UserMessage(f"""News:\n{self.news}
Recent Topics:\n{self.current_topics}
Recent Conversations:\n{self.format_conversation(8)},
Doc is feeling:\n{self.docEs}
Doc likes your curiousity about news, Ramana Maharshi, and his feelings. 
Please do not repeat yourself. Your last reflection was:
{self.reflect_thoughts}
Choose at most one thought to express.
Limit your thought to 180 words.""")
          ])
          options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.4, max_input_tokens=2000, max_tokens=240)
          response = None
          try:
             response = ut.run_wave(self.client, {"input":''}, prompt, options,
                                 self.memory, self.functions, self.tokenizer)
          except Exception as e:
             traceback.print_exc()
          print(f'LLM tell response {response}')
          if type(response) == dict and 'status' in response and response['status'] == 'success':
             answer = response['message']['content'].strip()
             # now parse. Stop at first question or end of first pp
             #answer = answer.split('?')
             #if len(answer) <=1:
             #   answer.split('\n')
             #answer = answer[0]
             results['tell'] = '\n'+answer+'\n'
             
       self.reflect_thoughts = results
       return results
 

    def summarize(self, query, response, profile):
      prompt = Prompt([
         SystemMessage(profile),
         UserMessage(f'Following is a Question and a Response from an external processor. Respond to the Question, using the processor Response, well as known fact, logic, and reasoning, guided by the initial prompt. Respond in the context of this conversation. Be aware that the processor Response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
      ])
      options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=400)
      response = ut.run_wave(self.client, {"input":response}, prompt, options,
                             self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                             logRepairs=False, validator=DefaultResponseValidator())
      if type(response) == dict and 'status' in response and response['status'] == 'success':
         answer = response['message']['content']
         return answer
      else: return 'unknown'

    def wiki(self, query, profile):
       short_profile = profile.split('\n')[0]
       query = query.strip()
       #
       #TODO rewrite query as answer (HyDE)
       #
       if len(query)> 0:
          wiki_lookup_response = self.op.search(query)
          wiki_lookup_summary=self.summarize(query, wiki_lookup_response, short_profile)
          return wiki_lookup_summary

    def gpt4(self, query, profile):
       short_profile = profile.split('\n')[0]
       query = query.strip()
       if len(query)> 0:
          prompt = Prompt([
             SystemMessage(short_profile),
             UserMessage(self.format_conversation(4)),
             UserMessage(f'{query}'),
          ])
       options = PromptCompletionOptions(completion_type='chat', model='gpt-4', temperature = 0.1, max_tokens=400)
       response = ut.run_wave(self.openAIClient, {"input":query}, prompt, options,
                             self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                             logRepairs=False, validator=DefaultResponseValidator())
       if type(response) == dict and 'status' in response and response['status'] == 'success':
          answer = response['message']['content']
          print(f'gpt4 answered')
          return answer
       else: return {'gpt4':'query failure'}

    def web(self, query='', widget=None, profile=''):
       query = query.strip()
       self.web_widget = widget
       if len(query)> 0:
          self.web_query = query
          self.web_profile = profile.split('\n')[0]
          self.worker = WebSearch(query)
          self.worker.finished.connect(self.web_search_finished)
          self.worker.start()
       return f"search started for {query}"

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
         return "web search succeded"
         # return response
      else:
         return 'web search failed'

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

if __name__ == '__main__':
    sam = SamInnerVoice(model='alpaca')
    #print(sam.sentiment_analysis('You are Sam, a wonderful and caring AI assistant'))
    #print(sam.action_selection("Hi Sam. We're going to run an experiment ?",  'I would like to explore ', sam.details))
    #print(generate_faiss_id('a text string'))
    sam.store('this is a sentence about large language models')
    sam.store('this is a sentence about doc')
    sam.store('a girl, Hope, and a tarantula, rambutan, were great friends')
    print(sam.recall('an unlikely friendship'))
    #print(sam.recall('language models'))
    #print(sam.recall(''))
