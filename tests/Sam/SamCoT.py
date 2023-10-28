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

host = '127.0.0.1'
port = 5004

def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

class LLM():
   def __init__(self, model='alpaca'):
        self.model = model
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':[], 'history':[]})

   def ask(self, client, input, prompt_msgs, temp=0.2, max_tokens=100, validator=DefaultResponseValidator()):
      """ Example use:
          class_prefix_prompt = [SystemMessage(f"Return a short camelCase name for a python class supporting the following task. Respond in JSON using format: {{"name": '<pythonClassName>'}}.\nTask:\n{form}")]
          prefix_json = self.llm.ask(self.client, form, class_prefix_prompt, max_tokens=100, temp=0.01, validator=JSONResponseValidator())
          print(f'***** prefix response {prefix_json}')
          if type(prefix_json) is dict and 'name' in prefix_json.keys():
             prefix = prefix_json['name']
      """

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
       
class SamInnerVoice():
    def __init__(self, model):
        self.client = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
        self.llm = LLM()
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.max_tokens = 4000
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.model = model
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.docEs = None # docs feelings
        self.current_topics = None # topics under discussion - mostly a keyword list
        self.last_tell_time = int(time.time()) # how long since last tell
        docHash_loaded = False
        try:
            self.docHash = {}
            with open('SamDocHash.pkl', 'rb') as f:
                data = pickle.load(f)
                self.docHash = data['docHash']
            docHash_loaded = True
        except Exception as e:
            # no docHash, so faiss is useless, reinitialize both
            self.docHash = {}
            with open('SamDocHash.pkl', 'wb') as f:
                data = {}
                data['docHash'] = self.docHash
                pickle.dump(data, f)
        # create wiki search engine
        self.op = op.OpenBook()
        self.action_selection_occurred = False

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

    def recall(self, topic, query, profile=None, history= None, retrieval_count=2, retrieval_threshold=.8):
        query_embed = self.embedder.encode(query)

        # gather docs matching tag filter
        candidate_ids = []
        vectors = []
        # gather all docs
        for entry in self.docHash:
           # gather all potential docs
           candidate_ids.append(entry['id'])
           vectors.append(self.docHash[id]['embed1'])
           vectors.append(self.docHash[id]['embed2'])

        # add all matching docs to index:
        index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
        index.add_with_ids(np.array(vectors), np.array(candidate_ids))
        distances, ids = index.search(query_embed.reshape(1,-1), min(10, len(candidate_ids)))
        print("Distances:", distances)
        print("Id:",ids)
        texts = []
        timestamps = [self.metaData[i]['timestamp'] for i in ids[0]]
        # Compute score combining distance and recency
        scores = []
        for dist, id, ts in zip(distances[0], ids[0], timestamps):
            age = (datetime.now() - ts).days
            score = dist + age * 0.1 # Weighted
            scores.append((id, score))
        # Sort by combined score
        results = sorted(scores, key=lambda x: x[1])
        
        texts = []
        for idx in range(len(results)):
           if len(texts) < retrieval_count:
              texts.append(self.docHash[results[idx][0]['text']])

        return str(texts)
    
    def store(self, text, profile=None, history=None):
       # opposite of recall
       # ask AI to generate two keys to store this under
       print(f'SamCoT store entry {text}')
       id = generate_faiss_id(text)
       if id in self.docHash:
          print('duplicate id, error, skipping')
          return False

       keys_prompt = [SystemMessage(f"""Generate two short descriptive text strings for the following content. Respond in JSON using format: {{"key1": '<a short descriptive string>',"key2":'<a content-orthogonal short descriptive string>'}}"""),
                      UserMessage(f'Content:\n{text}')]
       keys_json = self.llm.ask(self.client, text, keys_prompt, max_tokens=100, temp=0.3, validator=JSONResponseValidator())
       print(f' generated keys: {keys_json}')
       key1 = text; key2 = text
       if type(keys_json) is dict and 'key1' in keys_json.keys():
          key1 = keys_json['key1']
          embed1 = self.embedder.encode(key1)
       if type(keys_json) is dict and 'key2' in keys_json.keys():
          key2 = keys_json['key2']
          embed2 = self.embedder.encode(key2)
       self.docHash[id] = {"id":id, "text":text, "key1":key1, "embed1":embed1, "key2":key2, "embed2":embed2, "timestamp":datetime.now()}
       short_form={k: self.docHash[id][k] for k in ["key1", "key2", "text"]}
       print(f'Storing {short_form}')
       # and save - write docHash first, we can always recover from that.
       with open('SamDocHash.pkl', 'wb') as f:
          data = {}
          data['docHash'] = self.docHash
          pickle.dump(data, f)
             
    def get_all_wmkeys(self):
        wmkeys = []
        for entry in self.docHash:
            if 'key1' in entry:
                wmkeys.append(key1)
            if 'key2' in entry:
                wmkeys.append(key2)
        return wmkeys
                        

    def action_selection(self, input, profile, history, widget):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        print(f'action selection input {input}')
        self.action_selection_occurred = True
        short_profile = profile.split('\n')[0]
        self.articles = []
        for key in self.details.keys():
            for item in self.details[key]:
                self.articles.append({"title": item['title'] })
        prompt_text = f"""{short_profile}
Given the following user input, determine if an agent with the above profile should act at this time.
User input:
{input}

Look first for actions specifically requested in the input. 
If you need more information you can use the action 'ask'.
The usual default action should be to choose 'none'.

New York Times news headlines for today:
{self.articles}

full articles can be retrieved using the action 'article'.

Respond using the following TOML format:
[RESPONSE]
action="<action name>"
value= "<action argument>"
[STOP]


The actions available are:
action_name\naction_argument\tdescription\texample
none\t'none'\t no action is needed.\t[RESPONSE]\naction=none\nvalue=none\n[STOP]
ask\t<question>\t ask doc a question.\t[RESPONSE]\naction=ask\nvalue=how are you feeling, doc?\n[STOP]
article\t<article title>\t retrieve a NYTimes article.\t[RESPONSE]\naction=article\nvalue=To Combat the Opiod Epidemic, Cities Ponder Facilities for Drug Use\n[STOP]
gpt4\t<question>\t ask gpt4 a question\t[RESPONSE]\naction=gpt4\nvalue=in python on linux, how can I list all the subdirectory names in a directory?\n[STOP]
recall\t<key string>\t recall content from working, using the <key string> as the recall target.\t[RESPONSE]\naction=recall\nvalue=Cognitive Architecture>\n[STOP]
store\t<text to place in working memory> \t a text you want to remember that will be stored under embeddings for two AI generated key strings.\t[RESPONSE]\naction=store\nvalue=BoardState: {{"1a":' ',"1b":' ',"1c":' ',"2a":' ',"2b":' ',"2c":' ',"3a":' ',"3b":' ',"3c":' '}}\n[STOP]
web\t<search query string>\t perform a web search, using the <search query string> as the subject of the search.\t[RESPONSE]\naction=web\nvalue=Weather forecast for Berkeley,Ca for <today - eg Jan 1, 2023>\n[STOP]
wiki\t<search query string>\t search the local wikipedia database.\t[RESPONSE]\naction=wiki\nvalue=Want is the EPR paradox in quantum physics?\n[STOP]
"""
        action_validation_schema={
            "action": {
                "type":"string",
                "required": True,
                "meta": "<action to perform>"
            },
            "value": {
                "type":"string",
                "required": True,
                "meta": "argument for action"
            }
        }


        #print(f'action_selection {input}\n{response}')
        prompt = Prompt([
           SystemMessage(prompt_text),
           UserMessage(f'User Input:\n{{$input}}'),
        ])
        prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.2, max_tokens=50)

        text = self.format_conversation(history, 4)
        print(f'action_selection starting analysis')
        analysis = ut.run_wave (self.client, {"input": input}, prompt, prompt_options,
                              self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                              logRepairs=False, validator=TOMLResponseValidator(action_validation_schema))
      

        summary_prompt_text = f"""Summarize the information in the following text with respect to its title {{$title}}. Do not include meta information such as description of the content, instead, summarize the actual information contained."""

        summary_prompt = Prompt([
            SystemMessage(summary_prompt_text),
            UserMessage('{{$input}}'),
            #AssistantMessage(' ')
        ])
        summary_prompt_options = PromptCompletionOptions(completion_type='chat', model='gpt-3.5-turbo', max_tokens=240)

        print(f'action_selection analysis: {analysis}')
        if type(analysis) == dict and 'status' in analysis and analysis['status'] == 'success':
            content = analysis['message']['content']
            print(f'SamCoT content {content}')
            if type(content) == dict and 'action' in content.keys() and content['action']=='article':
                title = content['value']
                print(f'Sam wants to read article {title}')
                ok = self.confirmation_popup(f'Sam want to retrieve {title}')
                if not ok: return {"none": ''}
                article = self.search_titles(title)
                url = article['url']
                print(f' requesting url from server {title} {url}')
                response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}', timeout=15)
                data = response.json()
                summary = ut.run_wave(self.openAIClient, {"input":data['result']}, summary_prompt, summary_prompt_options,
                                      self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                      logRepairs=False, validator=DefaultResponseValidator())
                                      
                print(f'retrieved article summary:\n{summary}')
                if type(summary) is dict and 'status' in summary.keys() and summary['status']=='success':
                    return {"article":  summary['message']['content']}
                else:
                    return {"none": ''}
            elif type(content) == dict and 'action' in content.keys() and content['action']=='web':
                ok = self.confirmation_popup(content)
                if ok:
                   self.web(content['value'], widget, short_profile, history)
                   return {"web":content['value']}
            elif type(content) == dict and 'action' in content.keys() and content['action']=='wiki':
                ok = self.confirmation_popup(content)
                if ok:
                   found_text = self.wiki(content['value'], profile, history)
                   return {"wiki":found_text}
            elif type(content) == dict and 'action' in content.keys() and content['action']=='ask':
               ok = self.confirmation_popup(content)
               if ok:
                   return {"ask":content['value']}
            elif type(content) == dict and 'action' in content.keys() and content['action']=='gpt4':
                ok = self.confirmation_popup(content)
                if ok:
                   text = self.gpt4(content['value'], short_profile, history)
                   return {"gpt4":text}
            elif type(content) == dict and 'action' in content.keys() and content['action']=='recall':
                ok = self.confirmation_popup(content)
                if ok:
                   text = self.recall(content['value'], profile=short_profile, history=history)
                   return {"recall":text}
            elif type(content) == dict and 'action' in content.keys() and content['action']=='store':
                ok = self.confirmation_popup(content)
                if ok:
                   text = self.store(content['value'], profile=short_profile, history=history)
                   return {"store":text}
            else:
               return {"none": ''}

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


    def topic_analysis(self,profile_text, history):
       conversation = self.format_conversation(history)
       print(f'conversation\n{conversation}')
       keys_ents = self.get_keywords(conversation)
       print(f'keywords {keys_ents}')
       prompt = Prompt([SystemMessage("Assignment: Determine the main topics of a conversation based on the provided keywords below. The response should be an array of strings that represent the main topics discussed in the conversation based on the given keywords and named entities.\n"),
                        UserMessage('Keywords and Named Entities:\n{{$input}}\n.'),
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
                                      

    def reflect(self, profile_text, history):
       global es
       results = {}
       print('reflection begun')
       es = self.sentiment_analysis(profile_text)
       if es is not None:
          results['sentiment_analysis'] = es
       if self.current_topics is None:
          self.current_topics = self.topic_analysis(profile_text, history)
          print(f'topic-analysis {self.current_topics}')
          results['current_topics'] = self.current_topics
       now = int(time.time())
       if self.action_selection_occurred and now-self.last_tell_time > random.random()*60+60 :
          self.action_selection_occurred = False # reset action selection so no more tells till user input
          print('do I have anything to say?')
          self.last_tell_time = now
          prompt = Prompt([
             SystemMessage(profile_text.split('\n')[0:4]),
             UserMessage(f"""News:\n{self.news}
Recent Topics:\n{self.current_topics}
Recent Conversations:\n{self.format_conversation(history, 4)},
Doc is feeling:\n{self.docEs}
Doc likes your curiousity about news, Ramana Maharshi, and his feelings. 
Choose at most one thought to express.
Limit your thought to 200 words.""")
          ])
          options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.4, max_input_tokens=2000, max_tokens=300)
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
             
       return results
 

    def summarize(self, query, response, profile, history):
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

    def wiki(self, query, profile, history):
       short_profile = profile.split('\n')[0]
       query = query.strip()
       #
       #TODO rewrite query as answer (HyDE)
       #
       if len(query)> 0:
          wiki_lookup_response = self.op.search(query)
          wiki_lookup_summary=self.summarize(query, wiki_lookup_response, short_profile, history)
          return wiki_lookup_summary

    def gpt4(self, query, profile, history):
       short_profile = profile.split('\n')[0]
       query = query.strip()
       if len(query)> 0:
          prompt = Prompt([
             SystemMessage(short_profile),
             UserMessage(self.format_conversation(history, 4)),
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

    def web(self, query='', widget=None, profile='', history=[]):
       query = query.strip()
       self.web_widget = widget
       if len(query)> 0:
          self.web_query = query
          self.web_profile = profile.split('\n')[0]
          self.web_history = history
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
            response = self.summarize(self.web_query, search_result['result']+'\n', self.web_profile, self.web_history)
         return response

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
    print(sam.sentiment_analysis('You are Sam, a wonderful and caring AI assistant'))
    print(sam.topic_analysis('You are Sam, a wonderful and caring AI assistant',
       [{"role": "### Response", "content": "How do you feel?"},
        {"role": "### Instruction", "content": "I feel ok, but I was asking about you.\n"},
        {"role": "### Instruction", "content": "\nHi Sam. Show me the most interesting (to me) news article of the day, please?\n"},
{"role": "### Response", "content": "Article summary:\nOpenAI is in talks"}
        ]))
          #print(sam.news)
    #print(sam.action_selection("Hi Sam. We're going to run an experiment ?",  'I would like to explore ', sam.details))
    #print(generate_faiss_id('a text string'))
    #sam.store('this is a sentence about large language models')
    #sam.store('this is a sentence about doc')
    #print(sam.recall('doc'))
    #print(sam.recall('language models'))
    #print(sam.recall(''))
