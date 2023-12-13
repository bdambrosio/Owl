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
from SamCoT import SamInnerVoice as Owl
from SamCoT import LLM as llm

today = date.today().strftime("%b-%d-%Y")

openai_api_key = os.getenv("OPENAI_API_KEY")
# List all available models
#try:
#    models = openai.Model.list()
#    for model in models.data:
#        print(model.id) 
#except openai.error.OpenAIError as e:
#    print(e)


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
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['January','February','March','April','May','June','July','August','September','October','November','December'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour

host = '127.0.0.1'
port = 5004

GPT4='gpt-4-1106-preview'
GPT35='gpt-3.5-turbo-1106'

def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

class Interactions:
    def __init__(self, template):
        self.template = template
        self.osClient = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.llm = llm(None, self.memory, self.osClient, self.openAIClient, self.template)
        self.owl = Owl(None, self.template)
        self.interactions = self.read_interaction_history('owl_ch.json')
        self.jsonValidator = JSONResponseValidator()
        self.max_tokens = 8000
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_names = self.owl.topic_names # keep this only in one place, we need to load SamCoT anyway.
        

    def read_interaction_history(self, file_path):
        interactions = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        interaction = json.loads(line)
                        interactions.append(interaction)
                    except json.JSONDecodeError:
                        print("Error decoding JSON from line:", line)
                        continue
        except Exception as e:
            print(f'Error loading interaction history {str(e)}')
        return interactions

    #
    ## Topic analysis, the main point of this file
    #

    def topic_analysis(self):
        for entry in self.owl.wm_topics:
            print(self.owl.wm_entry_printform(entry))
            
        updates = {}
        for i, interaction in enumerate(self.interactions):
            if type(interaction) is not dict or 'role' not in interaction:
               continue
            if interaction['role'] != 'user': # this will automatically skip assistant roles in building interactions
               continue
            user = self.interactions[i]
            # make sure we aren't at last entry, if not check if next entry is asst response
            if len(self.interactions) > i+1 and 'role' in self.interactions[i+1] and self.interactions[i+1]['role'] == 'assistant':
               asst = self.interactions[i]
            else:
               continue
            prompt = [SystemMessage(self.owl.short_prompt()),
                      UserMessage(f""""Determine the main topic of the following interaction, together with subtopics, key points, and a text summary. 
                      The topic must be one of:\n{self.owl.format_topic_names()}
                      The response should be a JSON form including the topic selected from above, possible subtopics, and key points: {{"topic": '<topic>', "subtopics":['<subtopic1>',], "key_points":'<a list of the key points to remember in this interaction>', "summary":'<a summary of the interaction with respect to the topic, subtopics, and key points>'}}
                      
                      Interaction:
                      {user}
                      {asst}
                      """),
                      AssistantMessage('')
                      ]

            update = self.llm.ask('', prompt, temp=0.1, max_tokens=125, validator = JSONResponseValidator())
            if update is not None:
                if type(update) is dict and 'topic' in update:
                    topic_name = 'Topic: '+update['topic']
                    if topic_name not in updates.keys():
                       updates[topic_name] = []
                    updates[topic_name].append(update)

        with open('topic_interaction_updates.json', 'w') as j:
           json.dump(updates, j)

        # process interaction updates for a topic into a single thread update for that topic
        for topic in updates.keys():
           print(f'{topic}, {len(updates[topic])}')
           #thread_update = self.create_thread_update(topic, updates[topic])
        return

    def create_thread_updates(self):
        with open('topic_interaction_updates.json', 'r') as j:
           interaction_updates = json.load(j)
         
        thread_updates = {}
        # create input
        for topic in interaction_updates.keys():
           thread = ""
           for update in interaction_updates[topic]:
              thread += json.dumps(update)+'\n'
           # specify prompt
           prompt = [SystemMessage(self.owl.short_prompt()),
                     UserMessage(f""""Combine the above individual interaction records into a single update record.
The response should be a JSON form, including topic, possible subtopics, key points, summary, inconsistencies, and future directions: 
{{"topic": '<topic>', 
  "subtopics":['<subtopic1>',], 
  "key_points":'<a list of the key points to remember in this interaction. key_points might be statements, hypotheses, todo item, questions, decisions, or commitments.>', 
  "summary":'<a text summary of the interaction with respect to the topic, subtopics, and key points>',
  "inconsistencies": '<a list of inconsistencies across the interaction summaries>',
  "future directions": '<a list of future directions indicated by the evolution of thought in the interaction summaries.>',
}}
                      
Subtopics, key_points, and summary should each be complete, accurate, yet concise compilations of the corresponding respective sets of interaction fields, without content duplication.

Interaction records:

{thread}

Respond only in JSON using the above format.
"""),
                     AssistantMessage('')
                     ]
        
           update = self.llm.ask('', prompt, temp=0.1, max_tokens=600, validator = JSONResponseValidator())
           print(f'{topic}:\n{update}')
           if update is not None:
              if type(update) is dict and 'topic' in update:
                 topic_name = 'Topic: '+update['topic']
                 thread_updates[topic_name] = update
              
        with open('topic_thread_updates.json', 'w') as j:
           json.dump(thread_updates, j)


    def update_topics(self):
        with open('topic_thread_updates.json', 'r') as j:
           topic_updates = json.load(j)
         
        # topic thread_updates uses fully qualified 'Topic: <topic_name>' as keys for topic items
        for topic in topic_updates.keys():
           print(f'\n\n***** processing topic {topic} *****')
           update = topic_updates[topic]
           docHash_container  = self.owl.find_wm_topic(topic) # find will add 'Topic: ' if needed
           print(f"    docHash found: {True if docHash_container is not None else False}")
           item = {"summary":'',
                   "subtopics": [],
                   "key_points":[],
                   "recent_thoughts":'',
                   "inconsistencies":[],
                   "future_directions":[]
                   }
           if docHash_container is None:
              # create new wm item
              print(f'     creating wm_topic {topic}')
              docHash_container = self.owl.create_awm(item, name=topic, notes=None, confirm=False)
              docHash_container  = self.owl.find_wm_topic(topic) # find will add 'Topic: ' if needed
              # older wm item, initialize w correct format
              print(f'initializing wm_topic {topic}')
              docHash_container['item']= item
              docHash_container['type'] = 'dict'
              self.owl.save_workingMemory()
              
           # ok, ready to update topic
           wm_topic = docHash_container['item']
           prompt = [SystemMessage(self.owl.short_prompt()+
                                   f"""
Your task is to update the current topic model below using the recent interaction summary. 
The current topic model is the more authoritative long-term representation, and is in JSON format as follows:
{{"summary":'',   # a comprehensive text summary of the questions, answers, and decisions in this topic
 "subtopics": [], # a list of major subtopics occuring in interactions
 "key_points":[], # a list of specific key information, statements, questions, decisions, and commitments on the topic or a subtopic.
 "recent_thoughts":'', # information, statements, questions, decisions, or commitments appearing for the first time. 
 "inconsistencies":[], # a list of inconsistencies noted across interactions 
 "future_directions":[] # a list of projections for future activity comparing recent thoughts to summary
}}
"""),
                     UserMessage(f"""Current topic model:
{json.dumps(wm_topic, indent=2)}

Recent interaction summary:

{json.dumps(update, indent=2)}

"""),
                     AssistantMessage('')
                     ]
           updated_wm_topic = self.llm.ask('', prompt, temp=0.1, max_tokens=600, stop_on_json=True, validator = JSONResponseValidator())
           print(f'   updated topic received from llm: {topic}\n{updated_wm_topic}\n')
           if updated_wm_topic is not None:
              if type(updated_wm_topic) is dict:
                 print(f'   updating wm_topic {topic}')
                 self.owl.update_wm_topic(docHash_container, updated_wm_topic)
              else: print(f'   update failed: updated_wm_topic type: {type(updated_wm_topic)}')
        print(f'saving wm_topics')
        self.owl.save_workingMemory()


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


if __name__ == '__main__':
    i = Interactions('zephyr')
    i.topic_analysis()
    i.create_thread_updates()
    i.update_topics()
