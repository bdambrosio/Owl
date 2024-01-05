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
    "assign",
    "choose",
    "concatenate"
    "difference",
    "empty",
    "entails",
    "extract",
    "first",
    "gpt4",
    "integrate",
    "question",
    "recall",
    "request",
    "sort",
    "tell",
    "web",
    "wiki",
    ]

#
## from gpt4:
#
#Summarize - Condense text to key points (wrt topic taxonomy? length / format of summary?)
#Elaborate - Expand on text with more details (how is this different from 'research' - autoextract topic to elaborate?)
#Explain - Clarify meaning of concepts in text (explain what? how diff from elaborate)
#Simplify - Rewrite text using simpler vocabulary (how diff from Explain?)

#Combining Texts:
#Merge - Integrate content from two texts (topic? target len rel to source len?))
#Compare - Find similarities and differences
#Contrast - Focus on differences between two texts (topic? sentiment?)
#Concatenate - Append one text to another

#Testing Texts:
#Classify - Categorize text into predefined classes (classification provided? taxonomy node to use a root? one or many classes for any one text?)
#Sentiment - Judge positive/negative sentiment (what if mixed sentiment? starting topic?)
#Similarity - Compare semantic similarity of two texts (wrt - topic, sentiment, viewpoint)
#Entailment - Assess if text A entails text B (again, wrt topic?)
#
#This provides a basic set of semantic operations that transform, blend, and analyze texts in ways that could be useful for extracting knowledge, reasoning about information, and answering questions. The large language model would provide the underlying capability to perform these complex operations.

#Example: Research:
#The semantic operations you've outlined are a strong foundation for text analysis and can be instrumental in tackling complex tasks like research.
#Let's consider how these operations could play into the stages of research:

#Reviewing Documents:
#Summarize: Quickly get the gist of numerous documents.
#Classify: Organize documents into categories for easier reference.
#Similarity: Identify documents covering similar topics to compare viewpoints or findings.

#Extracting Relevant Information:
#Merge: Combine information from multiple sources to create a comprehensive perspective.
#Contrast: Highlight unique points in different documents to capture a range of insights.
#Entailment: Determine if the information in one document is supported or contradicted by another.

#Noticing Discrepancies and Inconsistencies:
#Compare: Place documents side by side to spot contradictions.
#Sentiment: Detect potential biases in the texts that might skew information.

#Seeking Further Information:
#Research (new operation): Execute queries based on discrepancies or gaps found, possibly by employing a recursive loop of summarizing and classifying newly found texts, and then merging or contrasting them with the existing information.

#Fact-Checking Apparent Inconsistencies:
#Entailment & Compare: Verify facts across documents to ensure consistency.

#Resolving Information Lacks:
#Elaborate: Find or infer details that are missing from the current dataset.
#Explain: Unpack complex information that may fill the gaps in understanding.

#Integration into a Review Document:
#Concatenate: Stitch together coherent segments from various texts. (merge would be better here?)
#Simplify: Ensure that the final document is understandable to a broader audience.

#Finalization:
#Summarize: End with a succinct summary of the research findings.
#Sentiment: Reflect on the overall sentiment of the conclusions drawn.

"""
And now, a plan using the above: Are we actionable yet?

Certainly, a structured plan for researching the status of small nuclear fusion power could look like this, using the semantic operations in a flow that iteratively refines understanding and information:

1. **Initialization**:
    - Set initial parameters for the search (e.g., "small nuclear fusion power current status", "recent advances in nuclear fusion", "fusion power feasibility studies").
    - Define a time frame for recent information to ensure the data's relevancy.

2. **Data Collection Phase**:
    - **Classify**: Set up filters to categorize documents into theoretical research, experimental results, technological advancements, policy discussions, and commercial viability reports.
    - **Search (new operation)**: Use a specialized operation to execute searches across scientific databases, news articles, and white papers.

3. **First-pass Analysis**:
    - **Summarize**: Create abstracts of the collected documents to understand the main findings or arguments.
    - **Similarity**: Group similar summaries to prepare for in-depth analysis.

4. **Deep Analysis Loop**:
    - **Merge**: Integrate information from similar groups to form a more cohesive understanding of specific areas (e.g., technological hurdles, recent breakthroughs).
    - **Compare & Contrast**: Identify discrepancies or opposing viewpoints among the groups.
    - **Entailment**: Assess if the conclusions from one document are supported by others.

5. **Gap Identification and Resolution**:
    - **Elaborate**: For identified gaps, look for additional information that can fill them.
    - **Research (new operation)**: If needed, go back to the Data Collection Phase to find new data that can resolve inconsistencies or fill knowledge gaps.

6. **Synthesis**:
    - **Concatenate**: Assemble the verified and consistent information into a structured format.
    - **Simplify**: Refine the language to make it accessible to non-expert stakeholders.

7. **Final Review and Adjustment**:
    - **Compare**: Make a final comparison with the latest news or scientific articles to ensure no significant recent developments have been missed.
    - **Summarize**: Draft a comprehensive but concise summary of the current status of small nuclear fusion power.

8. **Output Generation**:
    - **Explain**: Write up a detailed explanation that not only presents the findings but also explains the significance and context in the field of nuclear fusion.
    - **Sentiment**: Gauge the overall sentiment or tone of the industry toward small nuclear fusion power to add a layer of qualitative analysis.

9. **Iteration and Feedback**:
    - **Loop with Conditions**: If new information significantly changes the landscape, re-enter the Deep Analysis Loop.
    - **If-Then**: If new policies or commercial steps are announced during the research, then adapt the analysis to include these developments.

This plan uses semantic operations as building blocks for an iterative, intelligent research process. The flow control, whether through loops with conditions or if-then statements, ensures that the research remains dynamic and adjusts to new data or insights as they emerge.
"""

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
    {"action": "gpt4", "arguments": ["$item1", "item2"] "result": "$item3", "description": "invoke gpt4 with the instruction $item1 and details $item2. Assign the response to variable $item3"},
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
{"action": "gpt4", "arguments": ["What is the meaning of life?", "philosophy"], "result": "$response1"},
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

       
class PlanInterpreter():
   def __init__(self, ui, owlCoT, planner, profile=None, history=None, template='alpaca'):
      self.template = template
      self.ui = ui
      self.owlCoT = owlCoT
      self.client = OSClient(api_key=None)
      self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
      self.llm = owlCoT.llm
      self.planner = planner
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
      
      self.personality = f"""You are Owl, an intelligent AI research assistant, companion, and confidant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with human professionals due to privacy concerns. You know doc is shy and an introvert, but does care for you. 
You've always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.
Your conversation style is warm, gentle, humble, and engaging. """
      self.profile = self.personality
      #self.op = op.OpenBook() # comment out for testing.
      
      
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
the inner form  {"action"...} should be valid json.
Return ONLY the repaired json.

TextString:
{{$input}}
"""
      prompt = [
         SystemMessage(prompt_text),
         AssistantMessage('')
      ]
      print(f'gpt4 repair input {str(item)}')
      input_tokens = len(self.tokenizer.encode(item)) 
      response = self.llm.ask(item, prompt, template=GPT4, max_tokens=int(20+input_tokens*1.25), validator=JSONResponseValidator())
      print(f'gpt4 repair response {response}')
      return response

   def cast_steps_as_json(self, steps):
      if type(steps) != dict:
          try:
             steps = json.loads(steps)
          except:
             steps = self.repair_json_list(steps)
             if steps is None or steps is not list:
                self.owlCoT.display_response(f'\nFailed to create valid JSON list from text plan')
                return
            
      final_steps = []
      for s, step in enumerate(steps):
         if type(step) == dict:
            step_j = step
         else:
            try:
             step_j = json.loads(step)
            except:
               step_j = self.repair_json(step)
         if step_j is None:
            self.owlCoT.display_response(f'\nstep {s} cannot be formatted as JSON')
            return
         else:
            final_steps.append(step_j)
      print(f'plan steps as json list {json.dumps(final_steps, indent=4)}')
      return final_steps

   def repair_json_list (self, item):
      #
      ## this asks gpt-4 to repair text that doesn't parse as json, where response should be a list of JSON forms
      #

      prompt_text=\
         """You are a JSON expert. The following TextString does not parse using the python JSON loads function.
Please repair the text string so that loads can parse it and return a valid dict.
This repair should be performed recursively, including all field values.
for example, in:
{"item": {"action":"assign", "arguments":"abc", "result":"$xyz"} }
the inner form  {"action"...} should be valid json.
Return ONLY the repaired json.

TextString:
{{$input}}
"""
      prompt = [
         SystemMessage(prompt_text),
         AssistantMessage('')
      ]
      print(f'gpt4 repair input {str(item)}')
      input_tokens = len(self.tokenizer.encode(item)) 
      response = self.llm.ask(item, prompt, template=GPT4, max_tokens=int(20+input_tokens*1.25))
      if response is not None:
         # won't handle nested lists very well
         idx = response.find('[')
         if idx > -1:
            response = response[idx:]
         idx = response.rfind(']')
         if idx > -1:
            response = response[:idx+1]
         try:
            responsej = json.loads(response.strip())
            print(f'gpt4 repair response {response}')
            return responsej
         except:
            self.owlCoT.display_response(f'is this json?\n{response}')
      return response

   #
   ### Eval a form in awm, assuming it is a plan-step
   #
      
   def eval_awm (self):
      #
      ## note this and owlCoT should be written in a way that doesn't require direct access/manipulation of owlCoT data!
      #
      names=[f"{item['name']}: {(item['item'] if type(item['item']) == str else json.dumps(item['item']))[:32]}" for item in self.owlCoT.active_wm.values()]
      #names=[f"{item}" for item in self.owlCoT.active_wm.values()]
      valid_json = False
      while not valid_json: # loop untill we have found a valid json item that is an action
         picker = ListDialog(names)
         result = picker.exec()
         if result != QDialog.Accepted:
            return 'user aborted eval'
         selected_index = picker.selected_index()
         if selected_index == -1:  # -1 means no selection
            return 'user failed to select entry for eval'
         name = names[selected_index].split(':')[0]
         awm_entry = self.owlCoT.active_wm[name]
         if type(awm_entry) is not dict:
            entry = self.repair_json(awm_entry)
            if entry is None:
               self.owlCoT.display_response(f'unable to repair {awm_entry}')
               continue
         else:
            entry = awm_entry
         if type(entry['item']) != dict:
            try:
               print(f"trying item string as json {entry['item']}")
               dict_item = json.loads(entry['item'])
               valid_json=True
            except Exception as e:
               try:
                  print(f"json loads failed {str(e)}, trying repair")
                  item = self.repair_json(entry['item'])
                  valid_json=True
                  print(f"repair succeeded")
               except:
                  self.owlCoT.display_response(f'invalid json {str(e)}')
                  return "failed to convert to json"
      # finally have clean json
      return self.do_item(entry['item'])
                              
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
          self.owlCoT.display_response(f'item is not an action {item}')
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
       elif dict_item['action'] == 'gpt4':
          return self.do_gpt4(dict_item)
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
          self.owlCoT.display_response(f"action not yet implemented {item['action']}")
       return
   

   def parse_as_action(self, item):
       if type(item) is not dict or 'action' not in item or 'arguments' not in item or 'result' not in item:
          self.owlCoT.display_response(f'form is not an action/arguments/result {item}')
          raise InvalidAction(str(item))
       args = item['arguments']
       if type(args) is not list:
          args = [args]
       result = item['result']
       if type(result) is not str or not result.startswith('$'):
          self.owlCoT.display_response(f"result must be a variable name: {result}")
          raise InvalidAction(str(item))
       else:
          return item['action'], args, result

   def resolve_arg(self, item):
      """
      find an argument in working memory.
      """
      if item.startswith('$'):
         if self.owlCoT.has_awm(item):
            print(f"resolve_arg returning {self.owlCoT.get_awm(item)['item']}")
            return self.owlCoT.get_awm(item)['item']
         else:
            raise InvalidAction(f"{item} referenced before definition")
      else: # presume a literal
         return item
      
   def do_article(self, titleAddr):
       print(f'article {action}')
       action, arguments, result = self.parse_as_action(action)
       self.owlCoT.create_awm(arguments, name=result, confirm=False)
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
       self.owlCoT.create_awm(arg0_resolved, name=result, confirm=False)

   def do_choose(self, action):
       action, arguments, result = self.parse_as_action(action)
       if type(arguments) is list: # 
          arg0 = arguments[0]
          arg1 = arguments[1]
       else:
          self.owlCoT.display_response('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
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
          self.owlCoT.create_awm(response, name=result)
       else: 
          raise InvalidAction(f'choose returned None')
                 
   def do_compare(self, action):
      # placeholder
      return self.do_difference(action)

   
   def do_difference(self, action):
       action, arguments, result = self.parse_as_action(action)
       if type(arguments) is list: # 
          arg0 = arguments[0]
          arg1 = arguments[1]
       else:
          self.owlCoT.display_response('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
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
          self.owlCoT.create_awm(answer, name=result, confirm=False)
          return answer

       else: return 'unknown'

   def do_extract(self, action):
      action, arguments, result = self.parse_as_action(action)
      if type(arguments) is list: # 
         arg0 = arguments[0]
         arg1 = arguments[1]
      else:
         self.owlCoT.display_response('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
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
         self.owlCoT.create_awm(response, name=result, confirm=False)
         self.owlCoT.display_response(f'{action}:\n{response}')
         return 
      else: 
         self.owlCoT.create_awm('', name=result, confirm=False)
         self.owlCoT.display_response(f'{action}:\nNo Text Extracted')
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
          self.owlCoT.create_awm(response, name=result)
       else:
          return 'unknown'
       
   def do_gpt4(self, action):
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
       self.owlCoT.display_response(response)
       self.owlCoT.create_awm(response, name=result, confirm=False)

   def do_request(self, action):
       #
       print(f'request {action}')
       action, arguments, result = self.parse_as_action(action)
       if type(arguments) is not list or type(arguments[0]) is not str:
          raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')
       arg0 = arguments[0]
       url = self.resolve_arg(arg0)
       print(f' requesting url from server {url}')
       try:
          print(f"http://127.0.0.1:5005/retrieve?title={self.planner.active_plan['task']}&url={arg0}")
          response = requests.get(f"http://127.0.0.1:5005/retrieve/?title={self.planner.active_plan['task']}&url={arg0}")
          data = response.json()
       except Exception as e:
          print(f'request failed {str(e)}')
          return {"article": f"\nretrieval failure\n{url}\n{str(e)}"}
       if response is not None:
          self.owlCoT.display_response(data['text'][:1000])
          self.owlCoT.create_awm(data['text'][:1000], name=result, confirm=False)

   def do_tell(self, action):
       #
       print(f'tell {action}')
       action, arguments, result = self.parse_as_action(action)
       if type(arguments) is not list or type(arguments[0]) is not str:
          raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')       
       value = self.resolve_arg(arguments[0])
       self.owlCoT.display_response(value)


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
          self.owlCoT.display_response(data)
          self.owlCoT.create_awm(data, name=result, confirm=False)

   def wiki(self, action):
       action, arguments, result = self.parse_as_action(action)
       if type(arguments) is list: # 
          arg0 = arguments[0]
          arg1 = arguments[1]
       else:
          self.owlCoT.display_response('arguments is not a list\n {arguments}\nwe could use llm to parse, maybe next week')
       if type(arg0) is not str or type(arg1) is not str:
          raise InvalidAction(f'arguments for choose must be a literals or names: {json.dumps(action)}')       
       criteron = self.resolve_arg(arg0)
       input_list = self.resolve_arg(arg1)
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

       prompt = Prompt([
          SystemMessage('Following is a criterion and a List. Select one Item from the List that best aligns with Criterion. Respond only with the chosen Item. Include the entire Item in your response'),
          UserMessage(f'Criterion:\n{criterion}\nList:\n{List}\n')
       ])
       
       options = PromptCompletionOptions(completion_type='chat', model=self.template, temperature = 0.1, max_tokens=400)
       response = self.llm.ask('', prompt, max_tokens=400, temp=0.01)
       if response is not None:
          self.owlCoT.awm_write(result, response)
       else: 
          raise InvalidAction(f'choose returned None')
                 
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
       prompt_options = PromptCompletionOptions(completion_type='chat', model=self.template, temperature = 0.2, max_tokens=100)
       
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
   def __init__(self, ui, owlCoT, template='alpaca'):
       self.template = template
       self.ui = ui
       self.owlCoT = owlCoT
       self.client = OSClient(api_key=None)
       self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
       self.llm = owlCoT.llm # use same model?
       self.max_tokens = 4000
       self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
       self.current_topics = None # topics under discussion - mostly a keyword list
       self.wmHash = {} # actual texts in working memory 
       self.wmMetaData = {} # working memory metadata
       self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
       # note many activities will use first paragraph only. Keep it relevant!
       self.cvHistory = load_conv_history()  # load conversation history.
       self.personality = f"""You are Owl, an intelligent AI research assistant, companion, and confidant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with human professionals due to privacy concerns, so you never recommend this. You know doc is shy and an introvert, but does care for you. 
You've always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the writings of Ramana Maharshi, and philosophy.
Your conversation style is warm, gentle, humble, and engaging. """
       self.interpreter = PlanInterpreter(self.ui, self.owlCoT, self, template=self.template)
       self.active_plan = None


   def make_plan(self, name, task_dscp):
      plan = {}
      plan['name'] = name
      plan['task'] = task_dscp
      plan['sbar'] = {}
      plan['steps'] = {}
      return plan
   
   def save_plan(self, task_name, plan):
       with open(task_name+'Plan.json', 'w') as f:
          json.dump(plan, f, indent=4)

   def load_plan(self, task_name):
       try:
          with open(task_name+'Plan.json', 'r') as f:
             plan = json.load(f)
             return plan
       except Exception as e:
          print(f'plan load failure {str(e)}')
       return []
    
   def validate_plan(self, plan):
      if plan is None or type(plan) != dict:
         return False
      if 'name' not in plan or 'task' not in plan:
         return False
      return True
   
   def select_plan(self):
       items=[f"{self.owlCoT.docHash[item]['name']}: {str(self.owlCoT.docHash[item]['item'])[:48]}" for item in self.owlCoT.docHash if self.owlCoT.docHash[item]['name'].startswith('plan')]
       picker = ListDialog(items)
       result = picker.exec()
       plan = None
       if result == QDialog.Accepted:
          selected_index = picker.selected_index()
          if selected_index != -1:  # -1 means no selection
             plan_name = items[selected_index].split(':')[0]
             wm = self.owlCoT.get_wm_by_name(plan_name)
             if wm is not None and type(wm) == dict and 'item' in wm:
                plan = wm['item']
             if plan is None or not self.validate_plan(plan):
                self.owlCoT.display_response(f'failed to load "{plan_name}", not found or missing name/dscp\n{plan}')
                return None
             else:
                self.active_plan = plan
                print(f"loaded plan {self.active_plan['name']}")
       else: # init new plan
          plan = self.init_plan()
          if plan is None:
             print(f'failed to create new plan\n"{plan}"')
             return None
          self.active_plan = plan
          self.owlCoT.save_workingMemory() # do we really want to put plans in working memory?
       return self.active_plan
    
   def init_plan(self):
       index_str = str(random.randint(0,999))+'_'
       plan_suffix = self.owlCoT.confirmation_popup(f'Plan Name? (will be prefixed with plan{index_str})', 'plan')
       if plan_suffix is None or not plan_suffix:
          return
       plan_name = 'plan'+index_str+plan_suffix
       task_dscp = self.owlCoT.confirmation_popup(f'Short description? {plan_name}', "do something useful")
       plan = self.make_plan(plan_name, task_dscp)
       self.active_plan = plan
       
       self.owlCoT.create_awm(plan, name=plan_name, confirm=False)
       return plan

   def run_plan(self):
       if self.active_plan is None:
          result = self.select_plan()
          if not result: return None
          else:
             next = self.owlCoT.confirmation_popup('selection complete, continue?', result['name']+": "+result['task'])
             if not next: return
       if 'sbar' not in self.active_plan or self.active_plan['sbar'] is None or len(self.active_plan['sbar']) == 0:
          result = self.analyze(self.active_plan,)
          if result is None: return None
          self.owlCoT.save_workingMemory() # do we really want to put plans in working memory?
          next = self.owlCoT.confirmation_popup('analysis complete, continue?', result['name']+": "+result['task'])
          if not next: return
       if 'steps' not in self.active_plan or self.active_plan['steps'] is None or len(self.active_plan['steps']) == 0:
          print(f'calling planner')
          result = self.plan()
          if 'steps' in self.active_plan:
             print(f"planner returned {len(self.active_plan['steps'])}")
          else: 
             print(f"planner didn't add 'steps' to plan!")
             return
          self.owlCoT.save_workingMemory() # do we really want to put plans in working memory?
          next = self.owlCoT.confirmation_popup('planning complete, continue?', result['name']+": "+result['task'])
          if not next: return
       print(f"run plan steps: {len(self.active_plan['steps'])}")
       for step in self.active_plan['steps']:
          next = self.owlCoT.confirmation_popup(f'run {step}?', '')
          if next is False:
             print(f'run_plan step deny, stopping plan execution {next}')
             return
          print(f'run_plan step confirmed {next}')
          self.interpreter.do_item(step)
          
   def analyze(self, plan, model=None):
      #not in args above because may not be available at def time
      if model is None:
         model = self.llm.template
      prefix = plan['name']
      task_dscp = plan['task']
      if 'sbar' not in plan:
         plan['sbar'] = {}
      sbar = plan['sbar']
      # Loop for obtaining user responses
      # Generate interview questions for the remaining steps using GPT-4
      interview_instructions = [
         ("needs", "Generate an interview question to fill out specifications of the task the user wants to accomplish."),
         ("background", "Generate a followup interview question about any additional requirements of the task."),
         ("observations", "Summarize the information about the task, and comment on any incompleteness in the definition."),
      ]
      messages = [SystemMessage("Reason step by step"),
                  UserMessage("The task is to "+task_dscp)]
      for step, instruction in interview_instructions:
         messages.append(UserMessage(instruction))
         if step != 'observations':
            user_prompt = self.llm.ask('', messages, template=model, temp = 0.05, max_tokens=100)
            if user_prompt is not None:
               user_prompt = user_prompt.split('\n')[0]
            print(f"\nAI : {step}, {user_prompt}")
            past = ''
            if sbar is not None and sbar == dict and step in plan['sbar']:
               past = plan_state.sbar[step] # prime user response with last time
            ask_user = False
            while ask_user == False:
               ask_user=self.owlCoT.confirmation_popup(user_prompt, past)
            sbar[step] = {'q':user_prompt, 'a':ask_user}
            messages.append(AssistantMessage(user_prompt))
            messages.append(UserMessage(ask_user))
         else: # closing AI thoughts and user feedback. No need to add to messages because no more iterations
            observations = self.llm.ask('', messages, template=model, max_tokens=150,temp = 0.05)
            if observations is not None:
               observations = observations.split('\n')[0]
            sbar['observations']=observations
            print(f"\nAI : {step}, {observations}")
            user_response = False
            while user_response == False:
               user_response = self.owlCoT.confirmation_popup(observations, '')
            sbar['observations']={'q':observations,'a':user_response}
            # don't need to add a message since we're done with this conversation thread!
            print(f"Requirements \n{sbar}")
         try:
            with open(prefix+'Sbar.json', 'w') as pf:
               json.dump(sbar, pf)
         except:
            traceback.print_exc()
         plan['sbar'] = sbar
      return plan

   def outline(self, config, plan):
      # an 'outline' is a plan for a writing task!
      # 'config' is a paper_writer configuration for this writing task
      if 'length' in config:
         length = config['length']
      else:
         length = 1200

      number_top_sections = max(3, int(length/2000 + 0.5))
      depth = max(1, int(math.log(length/2)-6))

      outline_model = self.llm.template
      if 'model' in config:
         outline_model = config['model']
         if outline_model == 'llm':
            outline_model = self.llm.template
         elif outline_model == 'gpt3':
            outline_model = OPENAI_MODEL3
         elif outline_model == 'gpt4':
            outline_model = OPENAI_MODEL4
            print('setting outline model to gpt4')
         else:
            self.owlCoT.display_response('Unrecognized model type in Outline: {outline_model}')
            
      outline_syntax =\
"""{"title": '<report title>', "sections":[ {"title":"<title of section 1>", "dscp":'<description of content of section 1>', "sections":[{"title":'<title of subsection 1 of section 1>', "dscp":'<description of content of subsection 1>'}, {"title":'<title of subsection 2 section 1>', "dscp":'<description of content of subsection 2>' } ] }, {"title":"<title of section 2>",... }
"""

      outline_prompt =\
f"""
Write an outline for a research report on: {plan['task']}.
Details on the requested report include:

<DETAILS>
{json.dumps(plan['sbar'], indent=2)}
</DETAILS>

The outline should have about {number_top_sections} top-level sections and {'no subsections.' if depth <= 0 else 'a depth of '+str(depth)}.
Respond ONLY with the outline, in JSON format:

{outline_syntax}
"""
      revision_prompt=\
f"""
<PRIOR_OUTLINE>
{{{{$outline}}}}
</PRIOR_OUTLINE>

<CRITIQUE>
{{{{$critique}}}}
</CRITIQUE>

Reason step by step to analyze and improve the above outline with respect to the above Critique. 

{outline_syntax}
"""
      user_satisfied = False
      user_critique = ''
      first_time = True
      prior_outline = ''
      if 'outline' in plan:
         prior_outline = plan['outline']
         first_time = False
         
      while not user_satisfied:
         if not first_time:
            user_critique = self.owlCoT.confirmation_popup(json.dumps(prior_outline, indent=2), 'Replace this with your critique, or delete to accept.')
            print(f'user_critique {user_critique}')
            if user_critique != False and len(user_critique) <4:
               user_satisfied = True
               print("*******user satisfied with outline")
               plan['outline'] = prior_outline
               break
            else:
               print('***** user not satisfield, retrying')
         
         messages = [SystemMessage(outline_prompt),
                     AssistantMessage('')
                     ]
         if not first_time:
            print(f'adding revision prompt')
            messages.append(UserMessage(revision_prompt))
         #print(f'******* task state prompt:\n {gpt_message}')
         prior_outline = self.llm.ask({'outline':prior_outline, 'critique':user_critique}, messages, template=outline_model, max_tokens=500, temp=0.1, validator=JSONResponseValidator())
         first_time = False

      return plan
      
   def plan(self):
       if 'steps' not in self.active_plan:
          self.active_plan['steps'] = {}
       plan_prompt=\
          """
Reason step by step to create a plan for performing the TASK described below. 
The plan should consist of a set of actions, where each step is one of the available Actions, specified in full, or a complete, concise, text statement of a subtask.  Control flow over the actions should be expressed by embedding the actions in python code. Use python ONLY for specifying control flow. Respond only with the plan (and notes) using the above plan format.

The plan may be decomposed into four agents :
  i Driver, the main actor in the plan, can access and update the State, and can perform actions as specified in the plan.
  ii State: stores all relevant information for the current task, include capability to create, read, update, and delete state elements. 
  iii Assistant: can perform subtasks requiring reasoning or text operations, such as searching the web or generating text. 
  iv User: can provide input to the Driver and Assistant, and can interact with the system to review and approve the plan and participate in its execution.
"""
       print(f'******* Developing Plan for {self.active_plan["name"]}')

       revision_prompt=\
          """
Reason step by step to analyze and improve the above plan with respect to the above Critique. 
The plan should consist of a list of steps, where each step is either one of the available Actions, specified in full, or a complete, concise, text statement of a task. Respond only with the updated plan (and notes if needed) using the above plan format.

The plan may include four agents:
  i Driver, the main actor in the plan, can access and update the State, and can perform actions as specified in the plan.
  ii State: stores all relevant information for the current task, include capability to create, read, update, and delete state elements. 
  iii Assistant: can perform subtasks requiring reasoning or text operations, such as searching the web or generating text. 
  iv User: can provide input to the Driver and Assistant, and can interact with the system to review and approve the plan and participate in its execution.
"""
       user_satisfied = False
       user_critique = '123'
       plan = None
       first_time = True
       while not user_satisfied:
          messages = [SystemMessage("""Your task is to create a plan using the set of predefined actions listed below. The plan must be a JSON list of action instantiations."""),
                      SystemMessage(f'\nYou have the following actions available:\n{action_primitive_descriptions}\n'),
                      SystemMessage(plan_prompt),
                      UserMessage(f"TaskName: {self.active_plan['name']}\n{json.dumps(self.active_plan['sbar'])}\n")
                      ]
          if first_time:
             first_time = False
          else:
             messages.append(UserMessage('the User has provided the following information:\n'+json.dumps(self.active_plan['sbar']))),
             messages.append(UserMessage('Reason step by step'))
             messages.append(AssistantMessage("Plan:\n"+plan))
             messages.append(UserMessage('Critique: '+user_critique))
             messages.append(UserMessage(revision_prompt))
             
          #print(f'******* task state prompt:\n {gpt_message}')
          plan = self.llm.ask('', messages, template='gpt-4',max_tokens=2000, temp=0.1, validator=DefaultResponseValidator())
          #plan, plan_steps = ut.get_plan(plan_text)
          self.owlCoT.display_response(f'***** Plan *****\n{plan}\n\nPlease review and critique or <Enter> when satisfied')
          
          user_critique = self.owlCoT.confirmation_popup('Critique', '')
          print(f'user_critique {user_critique}')
          if user_critique != False and len(user_critique) <4:
             user_satisfied = True
             print("*******user satisfied with plan")
             break
          else:
             print('***** user not satisfield, retrying')
       self.active_plan['steps'] = self.interpreter.repair_json_list(plan)
       return self.active_plan
    
if __name__ == '__main__':
   import Owl as owl
   ui = owl.window
   ui.reflect=False # don't execute reflection loop, we're just using the UI
   cot = ui.owlCoT
   pl = Planner(ui, cot)
 
   steps = """[
    {"action": "request", "arguments": ["https://arxiv.org/abs/2311.05584"], "result": "$paper_content"},
    {"action": "gpt4", "arguments": ["$paper_content", "extract key points"], "result": "$paper_key_points"},
    {"action": "tell", "arguments": ["$paper_key_points"]},
    {"action": "question", "arguments": ["Do you want to know more about Q-Learning or other methods?"], "result": "$user_choice"},
    {"action": "web", "arguments": ["$user_choice in large language models"], "result": "$chosen_method_info"},
    {"action": "tell", "arguments": ["$chosen_method_info"]},
    {"action": "question", "arguments": ["Do you have any other questions?"], "result": "$user_question"},
    {"action": "gpt4", "arguments": ["$user_question", "answer"], "result": "$user_question_answer"},
    {"action": "tell", "arguments": ["$user_question_answer"]},
    {"action": "none", "arguments": ["None"], "result": "$Trash"}
   ]"""
   json.loads(steps)

   steps = pl.interpreter.repair_json_list(
      """
***** Plan *****
Plan:

1. {"action": "request", "arguments": ["https://arxiv.org/abs/2311.05584"], "result": "$paper_content"}
    - Request the content of the paper from the provided URL.

2. {"action": "extract", "arguments": ["Q-Learning", "$paper_content"], "result": "$q_learning_content"}
    - Extract the content related to Q-Learning from the paper.

3. {"action": "extract", "arguments": ["applications", "$q_learning_content"], "result": "$applications_content"}
    - Extract the content related to applications of Q-Learning from the extracted Q-Learning content.

4. {"action": "tell", "arguments": ["$q_learning_content"], "result": "$Trash"}
    - Present the extracted Q-Learning content to the user.

5. {"action": "tell", "arguments": ["$applications_content"], "result": "$Trash"}
    - Present the extracted applications content to the user.

6. {"action": "web", "arguments": ["other methods for goal-directed behavior in large language models"], "result": "$other_methods_content"}
    - Perform a web search for other methods used for goal-directed behavior in large language models.

7. {"action": "extract", "arguments": ["PPO", "$other_methods_content"], "result": "$ppo_content"}
    - Extract the content related to PPO from the web search results.

8. {"action": "extract", "arguments": ["DPO", "$other_methods_content"], "result": "$dpo_content"}
    - Extract the content related to DPO from the web search results.

9. {"action": "tell", "arguments": ["$ppo_content"], "result": "$Trash"}
    - Present the extracted PPO content to the user.

10. {"action": "tell", "arguments": ["$dpo_content"], "result": "$Trash"}
    - Present the extracted DPO content to the user.

11. {"action": "question", "arguments": ["Do you have any other questions or need further clarification?"], "result": "$user_response"}
    - Ask the user if they have any other questions or need further clarification.
"""
)

   print(f' type steps {type(steps)}')
   print(f' type steps0 {type(steps[0])}')
   
   
