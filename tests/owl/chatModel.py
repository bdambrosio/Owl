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
from OwlCoT import OwlInnerVoice as oiv
from OwlCoT import LLM as llm
import rewrite as rw

today = date.today().strftime("%b-%d-%Y")

openai_api_key = os.getenv("OPENAI_API_KEY")
# List all available models
#try:
#    models = openai.Model.list()
#    for model in models.data:
#        print(model.id) 
#except openai.error.OpenAIError as e:
#    print(e)


local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['January','February','March','April','May','June','July','August','September','October','November','December'][month_num-1]
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

class Interactions:
    def __init__(self, template):
        self.template = template
        self.osClient = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.owl = oiv(None)
        
        self.interactions = self.read_interaction_history('owl_ch.json')
        self.jsonValidator = JSONResponseValidator()
        self.max_tokens = 8000
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_names = self.owl.topic_names # keep this only in one place, we need to load OwlCoT anyway.
        

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

    def determine_thread(self, user_input, current_thread_name, active_threads, conversation_history):
        # Prepare the prompt for the LLM
        prompt = "Instructions:\n"
        prompt += "1. Analyze the user input and determine the most relevant thread from among the active threads.\n"
        prompt += "2. If the user input introduces a new topic or thread, suggest a new thread-name.\n"
        prompt += """3. The thread-name response format is the thread_name, followed by a ':' followed by a short explanation, of no more than 10 words, of the thread-name. For example:

chat:informal friendly conversation with no specific goal

"""
        prompt += "4. Do not include any discursive or explanatory text.\n"
        prompt += "5. End your response with '\n</ThreadName>'."
    
        # Add active threads to the prompt
        prompt += "Active Threads:\n"
        for thread_name in active_threads.keys():
            prompt += f" - {thread_name}: {active_threads[thread_name]}\n"
    
        # Add conversation history to the prompt
        prompt += "\nRecent Conversation History, to aid in determining thread change:\n"
        for interaction in conversation_history:
            prompt += f" - {interaction['role']}: {interaction['content'].strip()}\n"
    
        prompt += f'\nCurrently active thread: {current_thread_name}\n'
        # Add instructions for the LLM
        prompt += f"User Input: {user_input}\n\n"
        
        # Ask the LLM to determine the thread
        llm_response = self.owl.llm.ask({}, [SystemMessage(prompt), AssistantMessage('<ThreadName>')],
                                        temp=0.1, eos='</ThreadName>', max_tokens=25)
    
        # Process the LLM's response
        llm_response = llm_response.strip()
    
        if llm_response in active_threads:
            # User input belongs to an existing thread
            return llm_response
        else:
            # User input introduces a new thread
            new_thread_name = llm_response
            return new_thread_name

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
            if interaction['role'] != 'user':
               # we include assistant response via lookahead four lines down 
               continue
            user = self.interactions[i]
            # make sure we aren't at last entry, if not check if next entry is asst response
            if len(self.interactions) > i+1 and 'role' in self.interactions[i+1] and self.interactions[i+1]['role'] == 'assistant':
               asst = self.interactions[i+1]
            else:
               continue
            prompt = [SystemMessage(f""""Determine the main topic of the following interaction, together with subtopics, key points, and a text summary. 
                      The topic must be one of:\n{self.owl.format_topic_names()}
                      The response should be a JSON form including the topic selected from above, possible subtopics, and key points: {{"topic": '<topic>', "subtopics":['<subtopic1>',], "key_points":'<a list of the key points to remember in this interaction>', "summary":'<a summary of the interaction with respect to the topic, subtopics, and key points>'}}
                      
                      Interaction:
                      {user}
                      {asst}
                      """),
                      AssistantMessage('')
                      ]

            update = self.llm.ask('', prompt, temp=0.1, max_tokens=150, validator = JSONResponseValidator())
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
           prompt = [SystemMessage(f""""Combine the following individual interaction records into a single update record.
The response should be a JSON form, including topic, possible subtopics, key points, summary, inconsistencies, and future directions: 
{{"topic": '<topic>', 
  "subtopics":['<subtopic1>',], 
  "key_points":'<a list of the key points to remember in this interaction. key_points can be statements, hypotheses, todo items, questions, decisions, or commitments related to the topic.>', 
  "summary":'<a text summary of the interaction with respect to the topic, subtopics, and key points>',
  "inconsistencies": '<a list of inconsistencies across the interaction summaries>',
  "future directions": '<a list of future directions indicated by the evolution of thought in the interaction summaries.>',
}}
                      
Subtopics, key_points, and summary should each be complete, accurate, detailed compilations of the corresponding respective sets of interaction fields, without content duplication.

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
           
           summary_prompt = [SystemMessage("""Your task is to update the prior summary below for '{{$topic}}':

{{$summary}}

Using the recent interaction summary:

{{$update}}

Your updated summary should be detailed and complete, including all information from the prior summary and updating it with the provided incremental information. Respond in this JSON format:

{"summary": '<a comprehensive text summary of the questions, answers, and decisions in this topic>'}
"""),
                     AssistantMessage('')
                     ]
           subtopics_prompt = [SystemMessage("""Your task is to update the prior topic subtopics:

{{$subtopics}}

Using the recent interaction subtopics:

{{$update}}
 
The updated subtopics list must use  this JSON format:

{"subtopics": ['subtopic-a', 'subtopic-b', ...]}
"""),
                     AssistantMessage('')
                     ]
           key_points_prompt = [SystemMessage("""Your task is to update the prior key_points:

{{$key_points}}

using the recent interaction key_points:

{{$update}}

Respond using this JSON format:

{"key_points": ['key_point-1', 'key_point-2', ...]}
"""),
                     AssistantMessage('')
                     ]

           summary=item['summary']
           summary_update = update['summary']
           updated_summary = self.llm.ask({"summary":summary, "update": summary_update}, summary_prompt, temp=0.1, max_tokens=1000, stop_on_json=True, validator = JSONResponseValidator())
           if updated_summary is not None:
              item['summary']=updated_summary
              
           subtopics = item['subtopics']
           subtopics_update = update['summary']
           updated_subtopics = self.llm.ask({"subtopics":subtopics, "update": subtopics_update}, subtopics_prompt, temp=0.1, max_tokens=1000, stop_on_json=True, validator = JSONResponseValidator())
           if updated_subtopics is not None:
              item['subtopics'] = updated_subtopics
              
           key_points = item['key_points']
           key_points_update = update['summary']
           updated_key_points = self.llm.ask({"key_points":key_points, "update":key_points_update},
                                             key_points_prompt,
                                             temp=0.1,
                                             max_tokens=1000,
                                             stop_on_json=True,
                                             validator = JSONResponseValidator())
           if updated_key_points is not None:
              item['key_points'] = updated_key_points

           print(f'   updated topic received from llm: {topic}\n{item}\n')
           if type(item) is dict:
              print(f'   updating wm_topic {topic}')
              self.owl.update_wm_topic(docHash_container, item)
           else: print(f'   update failed: updated_wm_topic type: {type(item)}')
       print(f'saving wm_topics')
       self.owl.save_workingMemory()


if __name__ == '__main__':
    interact = Interactions('zephyr')
    active_threads = {"chat":'general conversation not focused on a specific task or objective.',
                      "support":'user has asked for emotional support.'}
    for i in range(200):
        if interact.interactions[-i]['role'] != 'user':
            continue
        ch = interact.interactions[-(i+5):-i]
        current_thread_name='chat'
        thread_name = interact.determine_thread(interact.interactions[-i]['content'], current_thread_name, active_threads, ch).strip()
        thread_name = thread_name.replace('\\\n','')
        thread_parts = thread_name.split(':')
        thread_name = thread_parts[0].strip()
        thread_dscp=''
        if len(thread_parts)  > 1:
            thread_dscp = thread_parts[1].strip()
        if thread_name not in active_threads or len(active_threads[thread_name])<5 and len(thread_dscp) > 5:
            active_threads[thread_name]=thread_dscp
        print(f"{thread_name}: {thread_dscp}\n")
        current_thread_name = thread_name
    #i.update_topics()
    print(json.dumps(active_threads, indent=2))
