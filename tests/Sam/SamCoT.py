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
from datetime import datetime, date
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

class SamInnerVoice():
    def __init__(self, model):
        self.client = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key)

        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.max_tokens = 4000
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.model = model
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.docEs = None
        docHash_loaded = False
        try:
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'rb') as f:
                data = pickle.load(f)
                self.docHash = data['docHash']
                self.metaData = data['metaData']
            docHash_loaded = True
        except Exception as e:
            # no docHash, so faiss is useless, reinitialize both
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'wb') as f:
                data = {}
                data['docHash'] = self.docHash
                data['metaData'] = self.metaData
                pickle.dump(data, f)
        # create wiki search engine
        self.op = op.OpenBook()


    def confirmation_popup(self, action):
       msg_box = QMessageBox()
       msg_box.setWindowTitle("Confirmation")
       msg_box.setText(f"Can Sam perform {action}?")
       msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
       retval = msg_box.exec_()
       if retval == QMessageBox.Yes:
          return True
       elif retval == QMessageBox.No:
          return False

    
    def logInput(self, input):
        with open('SamInputLog.txt', 'a') as log:
            log.write(input.strip()+'\n')

    def search_titles(self, query):
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
        print(f'similar title {titles[most_similar]}')
        return articles[most_similar]
    
    def sentiment_analysis(self, profile_text):
       if self.docEs is not None:
          return None
       try:
          with open('SamInputLog.txt', 'r') as log:
             inputLog = log.read()

          #lines = inputLog.split('\n')
          lines = inputLog[-4000:]
          prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=150)
        
          analysis_prompt_text = f"""Analyze the input from Doc below for it's emotional tone, and respond with a few of the prominent emotions present. Note that the later lines are more recent, and therefore more indicitave of current state. Select emotions that best match the emotional tone of Doc's input. Remember that you are analyzing Doc's state, not your own."""

          analysis_prompt = Prompt([
             SystemMessage(profile_text),
             SystemMessage(analysis_prompt_text),
             UserMessage("Doc's input: {{$input}}\n"),
             #AssistantMessage(' ')
          ])
          analysis = ut.run_wave (self.client, {"input": lines}, analysis_prompt, prompt_options,
                                  self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                  logRepairs=False, validator=DefaultResponseValidator())

          if type(analysis) is dict and 'status' in analysis.keys() and analysis['status'] == 'success':
             es = analysis['message']['content']
             print(f'analysis {es}')
             return "Doc's emotional state analysis:\n"+es
       except Exception as e:
          traceback.print_exc()
          print(f' sentiment analysis exception {str(e)}')
       return None

    def sentiment_response(self):
       # breaking this out separately from sentiment analysis
       prompt_text = f"""Given who you are:\n{profile_text}\nand your analysis of doc's emotional state\n{es}\nWhat would you say to him? If so, pick only the one or two most salient emotions. Remember he has not seen the analysis, so you need to explicitly include the names of any emotions you want to discuss. You have only about 100 words.\n"""
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

    def recall(self, tags, query, retrieval_count=2, retrieval_threshold=.8):
        if tags =='cancel':
            return []
        if tags is None or len(tags) == 0:
            query_tags = ''
        else:
            query_tags = tags.lower().split(',')
        query_embed = self.embedder.encode(query)

        # gather docs matching tag filter
        candidate_ids = []
        vectors = []
        # gather all docs with matching tags
        for id in self.metaData.keys():
            # gather all potential docs
            if len(query_tags) == 0 or any(tag in query_tags for tag in self.metaData[id]['tags']):
                candidate_ids.append(id)
                vectors.append(self.metaData[id]['embed'])

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
        print(results)
        
        texts = []
        for idx in range(min(len(results), retrieval_count)):
            if (len(texts) < retrieval_count):
                texts.append(self.docHash[results[idx][0]])
        return texts
    
    def remember(self, tags, text):
        if tags =='cancel':
            return
        tags = tags.lower().split(',')
        embed = self.embedder.encode(text)
        id = generate_faiss_id(text)
        if id in self.docHash:
            print('duplicate, skipping')
        #self.semanticIDMap.add_with_ids(embed.reshape(1,-1), np.array([id]))
        self.docHash[id] = text
        self.metaData[id] = {"tags": tags, "embed": embed, "timestamp":datetime.now()}
        # and save - write docHash first, we can always recover from that.
        with open('SamDocHash.pkl', 'wb') as f:
            data = {}
            data['docHash'] = self.docHash
            data['metaData'] = self.metaData
            pickle.dump(data, f)

    def get_all_tags(self):
        tags = []
        for id in self.metaData.keys():
            if 'tags' in self.metaData[id]:
                tags = list(set(tags) | set(self.metaData[id]['tags']))
        return tags
                        

    def action_selection(self, input, profile, history):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        print(f'action selection profile len {len(profile)}')
        
        self.articles = []
        for key in self.details.keys():
            for item in self.details[key]:
                self.articles.append({"title": item['title'],
                                      #"abstract": item['abstract']
                                      })
        prompt_text = f"""Given the following user-provided recent exchange, your task is to determine if profile\n{profile}\n should act at this time. Only select an action if explicity called for by the user in the recent exchange, or use action <ask> if you need more information. Your priorities are as follows: 
1. Monitor doc's mental and emotional state. This can be inferred from analyzing his input, or using <ask> to ask doc directly how he is feeling.
2. Respond to doc's input.
3. Continue to explore your interest in the arts, sciences, and philosophy.

The usual default action should be to choose 'none'
The following New York Times articles are available:
{self.articles}

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
web\t<search query string>\t perform a web search, using the <search query string> as the subject of the search.\t[RESPONSE]\naction=web\nvalue=Weather forecast for Berkeley,Ca for <today's date>\n[STOP]
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
            UserMessage('Exchange:\n{{$input}}'),
        ])
        prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=50)

        text = self.format_conversation(history, 4)
        analysis = ut.run_wave (self.client, {"input": text}, prompt, prompt_options,
                              self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                              logRepairs=False, validator=TOMLResponseValidator(action_validation_schema))
      


        summary_prompt_text = f"""Summarize the information in the following text with respect to its title {{$title}}. Do not include meta information such as description of the content, instead, summarize the actual information contained."""

        summary_prompt = Prompt([
            SystemMessage(summary_prompt_text),
            UserMessage('{{$input}}'),
            #AssistantMessage(' ')
        ])
        summary_prompt_options = PromptCompletionOptions(completion_type='chat', model='gpt-3.5-turbo', max_tokens=240)

        print(f'SamCoT analysis {analysis}')
        if type(analysis) == dict and 'status' in analysis and analysis['status'] == 'success':
            content = analysis['message']['content']
            print(f'SamCoT content {content}')
            if type(content) == dict and 'action' in content.keys() and content['action']=='article':
                title = content['value']
                print(f'Sam wants to read article {title}')
                ok = self.confirmation_popup(f'Sam want to retrieve {title}')
                if not ok: return {"none": ''}
                article = self.search_titles(title, self.details)
                url = article['url']
                print(f' requesting url from server {title} {url}')
                response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}')
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
                   return {"gpt4":content['value']}
            else:
               return {"none": ''}

    def wakeup_routine(self):
        self.nytimes = nyt.NYTimes()
        self.news, self.details = self.nytimes.headlines()
        city, state = get_city_state()
        print(f"My city and state is: {city}, {state}")
        local_time = time.localtime()
        year = local_time.tm_year
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
        month_num = local_time.tm_mon
        month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
        month_day = local_time.tm_mday
        hour = local_time.tm_hour
        if hour < 12:
            return 'Good morning Doc!'
        if hour < 17:
            return 'Good afternoon Doc!'
        else:
            return 'Hi Doc.'
        # check news for anything interesting.
        # check todos
        # etc
        pass

    def get_keywords(self, text):
       prompt = Prompt([SystemMessage("Assignment: Extract keywords and named-entities from the conversation below.\nConversation:\n{{$input}}\n."),
                        #AssistantMessage('')
                        ])
       
       options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=50)
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
       options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=50)
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
       es = self.sentiment_analysis(profile_text)
       self.current_topics = self.topic_analysis(profile_text, history)
       print(f'topic-analysis {self.current_topics}')
       return es


    def summarize(self, query, response, profile, history):
      prompt = Prompt([
         SystemMessage(profile),
         UserMessage(f'Following is a Question and a Response from an external processor. Respond to the Question, using the processor Response, well as known fact, logic, and reasoning, guided by the initial prompt. Respond in the context of this conversation. Be aware that the processor Response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
      ])
      options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=400)
      response = ut.run_wave(self.client, {"input":response}, prompt, options,
                             self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                             logRepairs=False, validator=DefaultResponseValidator())
      if type(response) == dict and 'status' in response and response['status'] == 'success':
         answer = response['message']['content']
         return answer
      else: return 'unknown'

    def wiki(self, query, profile, history):
      query = query.strip()
      if len(query)> 0:
         wiki_lookup_response = self.op.search(query)
         wiki_lookup_summary=self.summarize(query, wiki_lookup_response, profile, history)
         return wiki_lookup_summary

    def web(self, query=None):
      query = query.strip()
      if len(query)> 0:
         self.web_query = query
         self.worker = WebSearch(selectedText)
         self.worker.finished.connect(self.web_search_finished)
         self.worker.start()
     
    def web_search_finished(self, search_result):
      if 'result' in search_result:
         response = ''
         if type(search_result['result']) == list:
            for item in search_result['result']:
               self.display_response('* '+item['source']+'\n')
               self.display_response('     '+item['text']+'\n\n')
               response += item['text']+'\n'
         elif type(search_result['result']) is str:
            self.display_response('\nWeb result:\n'+search_result['result']+'\n')
            self.add_exchange(self.web_query, response)
            
    

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
    #sam.remember('language models','this is a sentence about large language models')
    #sam.remember('doc', 'this is a sentence about doc')
    #print(sam.recall('doc','something about doc', 2))
    #print(sam.recall('language models','something about doc', 2))
    #print(sam.recall('','something about doc', 2))
