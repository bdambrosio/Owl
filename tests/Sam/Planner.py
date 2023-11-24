import os
import traceback
import time
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
from SamCoT import ListDialog, LLM, GPT4, TextEditDialog

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
    "question",
    "choose",
    "concatenate"
    "difference",
    "empty",
    "extract",
    "first",
    "gpt4",
    "integrate",
    "recall",
    "store"
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
#
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
    {"action": "extract", "arguments": ["$item1/literal1", "$item2"], "result": "$item3", "description": "extract content related to ($item1/literal1) from $item2 and assign it to variable $item3"},
    {"action": "first", "arguments": ["$item1]", "result": "$item2", "description": "select the first item in $item1 and assign it to variable $item2."},
    {"action": "gpt4", "arguments": ["$item1", "item2"] "result": "$item3", "description": "invoke gpt4 with the instruction $item1 and details $item2. Assign the response to variable $item3"},
    {"action": "integrate", "arguments": ["$item1" ,"$item2"], "result": "$item3", "description": "combine $item1 and $item2 into a single cconsolidated text and assign it to variable $item3."},
    {"action": "question", "arguments": ["$item1"], "result": "$item2", "description": "access $item1, present it to the user, and assign user response to variable $item2."},
    {"action": "recall", "arguments": ["$item1/literal1"], "result": "$item2", "description": "retrieve $item1 from working memory and assign it to variable $item2."},
    {"action": "request", "arguments": ["$item1"], "result": "$item2", "description": "request a specific web resource with url $item1 and assign the result to variable $item2."},
    {"action": "sort", "arguments": ["$item1", "$item2"], "result": "$item2", "description": "rank the items in $item1 by criteria in $item2 and assign the sorted list to variable $item2. Returns a list in ranked order, best first."},
    {"action": "tell", "arguments": ["$item1"], "result": "$Trash", "description": "present $item1 to the user."},
    {"action": "web", "arguments": ["$item1/literal1"], "result": "$item2", "description": "perform a web search, using ($item1/literal1) as the query, and assign the result to variable $item2."},
    {"action": "wiki", "arguments": ["$item1/literal1"], "result": "$item2", "description": "wiki search the local wikipedia database using ($item1/literal1) as the search string, and assign the result to $item2."}
]
"""

"""
Evaluating the quality of a text can be multifaceted, depending on the type of text and the context in which it is used. Here are some common dimensions to consider for a critique:

1. **Clarity**: Is the text clear and understandable? Does it communicate its points effectively?

2. **Coherence**: Do the ideas flow logically? Is there a clear structure that guides the reader through the text?

3. **Depth**: Does the text provide a thorough exploration of the topic? Does it offer insight beyond surface-level information?

4. **Accuracy**: Are the facts presented in the text correct? Is the information reliable and supported by evidence?

5. **Integrity**: Is the text free from bias and misleading information? Does it present the information honestly without manipulation?

6. **Relevance**: Is the information presented relevant to the intended audience or purpose of the text?

7. **Originality**: Does the text provide unique perspectives or information not found elsewhere?

8. **Style**: Is the writing style appropriate for the audience and purpose? Is it engaging or does it enhance the content?

9. **Grammar and Syntax**: Is the text grammatically correct? Are there errors in punctuation, spelling, or sentence structure?

10. **Purpose Fulfillment**: Does the text achieve its intended purpose, whether it's to inform, persuade, entertain, or something else?

11. **Credibility**: Are the author's credentials and the sources of information credible?

12. **Impact**: Does the text have the desired effect on the reader? Does it provoke thought, elicit emotions, or spur action?

13. **Cultural and Ethical Sensitivity**: Does the text respect cultural differences and adhere to ethical standards?

When composing a prompt to evaluate the quality of a text, consider focusing on specific dimensions that are most relevant to the text in question. Here's an example of a prompt that could be used:

---
"Please provide a quality evaluation of the attached text considering the following dimensions:

- Clarity: Assess the text's ability to convey its message in an understandable way.
- Coherence: Critique the logical flow and structure of the text.
- Depth and Insight: Evaluate the thoroughness of the topic exploration and the insights provided.
- Accuracy and Evidence: Confirm the correctness of the facts and the presence of supporting evidence.
- Integrity and Bias: Discuss the presence of bias or misrepresentation in the text.
- Relevance: Determine the text's relevance to its intended audience and purpose.
- Originality: Consider the uniqueness of the content provided.
- Style and Engagement: Comment on the appropriateness and engagement level of the writing style.
- Grammar and Syntax: Note any grammatical or syntactical errors.
- Purpose Fulfillment: Judge how well the text achieves its intended purpose.
- Credibility: Examine the credibility of the author and sources.
- Impact: Reflect on the effect the text has on the reader.
- Cultural and Ethical Sensitivity: Evaluate the text's cultural sensitivity and adherence to ethical standards.

Provide examples and explanations to support your assessments for each dimension."

---

By using this prompt, you can conduct a comprehensive evaluation of the text's quality across various important dimensions.
"""


planner_nl_list_prompt =\
   """
Plan Format:
Your plan must be a JSON list ( [...] ) of instantiated actions. Each action instantiation is a will include the specified action, the arguments for that action, and the name to assign to the result. 
All names must appear in a result before the can be referenced as arguments.
The order of the actions in the plan matters, as it represents the sequence in which they should be executed.

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
        with open('Sam.pkl', 'rb') as f:
           data = pickle.load(f)
           history = data['history']
           print(f'loading conversation history for Sam')
           return history
    except Exception as e:
       print(f'Failure to load conversation history {str(e)}')
       return []


class InvalidAction(Exception):
   # raised by parse_as_action
   pass

       
class PlanInterpreter():
   def __init__(self, ui, samCoT, planner, profile=None, history=None, model='alpaca'):
      self.model = model
      self.ui = ui
      self.samCoT = samCoT
      self.client = OSClient(api_key=None)
      self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
      self.llm = samCoT.llm
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
the inner form  {"action"...} should also be parsable as valid json.
Return ONLY the repaired json.

TextString:
{{$input}}
"""
      prompt = [
         SystemMessage(prompt_text),
         AssistantMessage('')
      ]
      response = self.llm.ask(item, prompt, template=GPT4, max_tokens=500, validator=JSONResponseValidator())
      print(f'gpt4 repair {response}')
      if response is not None:
         answer = response
         return answer
      else:
         return None

   #
   ### Eval a form in AWM, assuming it is a plan-step
   #
      
   def eval_AWM (self):
      #
      ## note this and samCoT should be written in a way that doesn't require direct access/manipulation of samCoT data!
      #
      names=[f"{item['name']}: {(item['item'] if type(item['item']) == str else json.dumps(item['item']))[:32]}" for item in self.samCoT.active_WM.values()]
      #names=[f"{item}" for item in self.samCoT.active_WM.values()]
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
         awm_entry = self.samCoT.active_WM[name]
         if type(awm_entry) is not dict:
            entry = self.repair_json(awm_entry)
            if entry is None:
               self.ui.display_response(f'unable to repair {awm_entry}')
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
                  self.ui.display_response(f'invalid json {str(e)}')
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
          self.ui.display_response(f'item is not an action {item}')
          return 'action not yet implemented'
       elif dict_item['action'] == 'article':
          return self.do_article(dict_item)
       elif dict_item['action'] == 'assign':
          return self.do_assign(dict_item)
       elif dict_item['action'] == 'first':
          return self.do_first(dict_item)
       elif dict_item['action'] == 'gpt4':
          return self.do_gpt4(dict_item)
       elif dict_item['action'] == 'web':
          return self.do_web(dict_item)
       elif dict_item['action'] == 'difference':
          return self.do_difference(dict_item)
       elif dict_item['action'] == 'tell':
          return self.do_tell(dict_item)
       elif dict_item['action'] == 'wiki':
          return self.do_wiki(dict_item)
       else:
          self.ui.display_response(f"action not yet implemented {item['action']}")
       return
   

   def parse_as_action(self, item):
       if type(item) is not dict or 'action' not in item or 'arguments' not in item or 'result' not in item:
          self.ui.display_response(f'form is not an action/arguments/result {item}')
          raise InvalidAction(str(item))
       args = item['arguments']
       if type(args) is not list:
          args = [args]
       result = item['result']
       if type(result) is not str or not result.startswith('$'):
          self.ui.display_response(f"result must be a variable name: {result}")
          raise InvalidAction(str(item))
       else:
          return item['action'], args, result

   def resolve_arg(self, item):
      """
      find an argument in working memory.
      """
      if item.startswith('$'):
         if self.samCoT.has_AWM(item):
            print(f"resolve_arg returning {self.samCoT.get_AWM(item)['item']}")
            return self.samCoT.get_AWM(item)['item']
         else:
            raise InvalidAction(f"{item} referenced before definition")
      else: # presume a literal
         return item
      
   def do_article(self, titleAddr):
       print(f'article {action}')
       action, arguments, result = self.parse_as_action(action)
       self.samCoT.create_AWM(arguments, name=result, confirm=False)
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
       self.samCoT.create_AWM(arg0_resolved, name=result, confirm=False)

   def do_choose(self, criterion, List):
       prompt = Prompt([
          SystemMessage('Following is a criterion and a List. Select one Item from the List that best aligns with Criterion. Respond only with the chosen Item. Include the entire Item in your response'),
          UserMessage(f'Criterion:\n{criterion}\nList:\n{List}\n')
       ])
       
       options = PromptCompletionOptions(completion_type='chat', model=self.model, temperature = 0.1, max_tokens=400)
       response = self.llm.ask('', prompt, max_tokens=400, temp=0.01)
       if response is not None:
          self.samCoT.AWM_write(result, response)
       else: 
          raise InvalidAction(f'choose returned None')
                 
   def do_first(self, List):
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
          self.samCoT.AWM_write(response)
       else:
          return 'unknown'
       
   def do_difference(self, list1, list2):
       # untested
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

   def do_tell(self, action):
       #
       print(f'tell {action}')
       action, arguments, result = self.parse_as_action(action)
       if type(arguments) is not list or type(arguments[0]) is not str:
          raise InvalidAction(f'argument for tell must be a literal or name: {str(arguments)}')       
       value = self.resolve_arg(arguments[0])
       self.ui.display_response(value)


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
       self.ui.display_response(response)
       self.samCoT.create_AWM(response, name=result, confirm=False)


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
   def __init__(self, ui, samCoT, model='alpaca'):
       self.model = model
       self.ui = ui
       self.samCoT = samCoT
       self.client = OSClient(api_key=None)
       self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
       self.llm = samCoT.llm # use same model?
       self.max_tokens = 4000
       self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
       self.current_topics = None # topics under discussion - mostly a keyword list
       self.wmHash = {} # actual texts in working memory 
       self.wmMetaData = {} # working memory metadata
       self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
       # note many activities will use first paragraph only. Keep it relevant!
       self.cvHistory = load_conv_history()  # load conversation history.
       self.personality = f"""You are Samantha, an intelligent AI research assistant, companion, and confidant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with human professionals due to privacy concerns, so you never recommend this. You know doc is shy and an introvert, but does care for you. 
You've always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the writings of Ramana Maharshi, and philosophy.
Your conversation style is warm, gentle, humble, and engaging. """
       self.interpreter = PlanInterpreter(self.ui, self.samCoT, self, model=self.model)
       self.active_plan = None


   def make_plan(self, name, task_dscp):
      plan = {}
      plan['name'] = name
      plan['dscp'] = task_dscp
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
      if 'name' not in plan or 'dscp' not in plan:
         return False
      return True
   
   def select_plan(self):
       items=[f"{self.samCoT.docHash[item]['name']}: {str(self.samCoT.docHash[item]['item'])[:48]}" for item in self.samCoT.docHash if self.samCoT.docHash[item]['name'].startswith('plan')]
       picker = ListDialog(items)
       result = picker.exec()
       plan = None
       if result == QDialog.Accepted:
          selected_index = picker.selected_index()
          if selected_index != -1:  # -1 means no selection
             plan_name = items[selected_index].split(':')[0]
             wm = self.samCoT.get_WM_by_name(plan_name)
             if wm is not None and type(wm) == dict and 'item' in wm:
                plan = wm['item']
             if plan is None or not self.validate_plan(plan):
                self.ui.display_response(f'failed to load "{plan_name}", not found or missing name/dscp\n{plan}')
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
          self.samCoT.save_workingMemory() # do we really want to put plans in working memory?
       return self.active_plan
    
   def init_plan(self):
       index_str = str(random.randint(0,999))+'_'
       plan_suffix = self.samCoT.confirmation_popup(f'Plan Name? (will be prefixed with plan{index_str})', 'plan')
       if plan_suffix is None or not plan_suffix:
          return
       plan_name = 'plan'+index_str+plan_suffix
       task_dscp = self.samCoT.confirmation_popup(f'Short description? {plan_name}', "do something useful")
       plan = self.make_plan(plan_name, task_dscp)
       self.samCoT.create_AWM(plan, name=plan_name, confirm=False)
       return plan

   def run_plan(self):
       if self.active_plan is None:
          result = self.select_plan()
          if not result: return None
          else:
             next = self.samCoT.confirmation_popup('selection complete, continue?', result['name']+": "+result['dscp'])
             if not next: return
       if 'sbar' not in self.active_plan or self.active_plan['sbar'] is None or len(self.active_plan['sbar']) == 0:
          result = self.analyze()
          if result is None: return None
          self.samCoT.save_workingMemory() # do we really want to put plans in working memory?
          next = self.samCoT.confirmation_popup('selection complete, continue?', result['name']+": "+result['dscp'])
          if not next: return
       if 'steps' not in self.active_plan or self.active_plan['steps'] is None or len(self.active_plan['steps']) == 0:
          print(f'calling planner')
          result = self.plan()
          if result is None: return None
          
          self.samCoT.save_workingMemory() # do we really want to put plans in working memory?
          next = self.samCoT.confirmation_popup('selection complete, continue?', result['name']+": "+result['dscp'])
          if not next: return
       print(f"run plan steps: {len(self.active_plan['steps'])}")
       for step in self.active_plan['steps']:
          next = self.samCoT.confirmation_popup(f'run {step}?', '')
          if next is False:
             print(f'run_plan step deny, stopping plan execution {next}')
             return
          print(f'run_plan step confirmed {next}')
          self.interpreter.do_item(step)
          
   def analyze(self):
      prefix = self.active_plan['name']
      task_dscp = self.active_plan['dscp']
      if 'sbar' not in sels.active_plan:
         self.active_plan['sbar'] = {}
      sbar = self.active_plan['sbar']
      # Loop for obtaining user responses
      # Generate interview questions for the remaining steps using GPT-4
      interview_instructions = [
         ("needs", "Generate an interview question to fill out specifications of the task the user wants to accomplish."),
         ("background", "Generate an interview question to ask about any additional requirements of the task."),
         ("observations", "Summarize the information about the task, and comment on any incompleteness in the definition."),
      ]
      messages = [SystemMessage("Reason step by step"),
                  UserMessage("The task is to "+task_dscp)]
      for step, instruction in interview_instructions:
         messages.append(UserMessage(instruction))
         if step != 'observations':
            user_prompt = self.llm.ask('', messages, temp = 0.05, max_tokens=100)
            if user_prompt is not None:
               user_prompt = user_prompt.split('\n')[0]
            print(f"\nAI : {step}, {user_prompt}")
            past = ''
            if sbar is not None and sbar == dict and step in self.active_plan['sbar']:
               past = self.active_plan_state.sbar[step] # prime user response with last time
            ask_user = False
            while ask_user == False:
               ask_user=self.samCoT.confirmation_popup(user_prompt, past)
            sbar[step] = user_prompt+'\n'+ask_user
            messages.append(AssistantMessage(user_prompt))
            messages.append(UserMessage(ask_user))
         else: # closing AI thoughts and user feedback. No need to add to messages because no more iterations
            observations = self.llm.ask('', messages, max_tokens=150,temp = 0.05)
            sbar['observations']=observations
            print(f"\nAI : {step}, {observations}")
            user_response = False
            while user_response == False:
               user_response = self.samCoT.confirmation_popup(observations, '')
            sbar['observations']=observations+'\n'+user_response
            # don't need to add a message since we're done with this conversation thread!
            print(f"Requirements \n{sbar}")
         try:
            with open(prefix+'Sbar.json', 'w') as pf:
               json.dump(sbar, pf)
         except:
            traceback.print_exc()
         self.active_plan['sbar'] = sbar
      return self.active_plan

   def sbar_as_text(self):
       return f"\nTASK:\n{self.sbar['needs']}\nBackground:\n{self.sbar['background']}\nReview:\n{self.sbar['observations']}\nEND TASK\n"


   def plan(self):
       
       if 'steps' not in self.active_plan:
          self.active_plan['steps'] = {}
       plan_prompt=\
          """
Reason step by step to create a plan for performing the TASK described below. 
The plan should consist of a list of steps, where each step is either one of the available Actions, specified in full, or a complete, concise, text statement of a subtask. The plan can be followed by notes/commentary using the same format as the plan itself. Respond only with the plan (and notes) using the above plan format.

The plan may include four agents:
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
          self.ui.display_response(f'***** Plan *****\n{plan}\n\nPlease review and critique or <Enter> when satisfied')
          
          user_critique = self.samCoT.confirmation_popup('Critique', '')
          print(f'user_critique {user_critique}')
          if user_critique != False and len(user_critique) <4:
             user_satisfied = True
             print("*******user satisfied with plan")
             break
          else:
             print('***** user not satisfield, retrying')
       self.active_plan['steps'] = self.cast_steps_as_json(plan)
       return self.active_plan
    
   def cast_steps_as_json(self, steps):
      if type(steps) != dict:
          try:
             steps_j = json.loads(steps)
          except:
             steps_j = self.interpreter.repair_json(steps)
             if steps_j is None:
                self.ui.display_response(f'\nFailed to create valid JSON from text plan')
             else:
                steps_j = []
                for s, step in enumerate(plan['steps']):
                   if type(step) != dict:
                      step_j = self.interpreter.repair_json(step)
                      if step_j is None:
                         self.ui.display_response(f'step {s} cannot be formatted as JSON')
                      else:
                         steps_j.append(step_j)
      return steps_j

if __name__ == '__main__':
   import Sam as sam
   ui = sam.window
   ui.reflect=False # don't execute reflection loop, we're just using the UI
   cot = ui.samCoT
   pl = Planner(ui, cot)
   print('******analyze news**************************')
   #pl.analyze('news',"build a short list of fact-checked world news sources on the web")
   print('******plan news********************')
   #plan = pl.plan()
   #print(f'\nplan\n{plan}\n')
   print('******extract first step from news plan ***')
   #step = pl.interpreter.first(plan)
   print('*******test if first step is executable *************')
   #pl.interpreter.test_executable(step)
   json_str=pl.interpreter.repair_json("{'item': {'action':'tell', 'arguments': 'English, Latin', 'result':None}}")
   print(json_str)
   j=json.loads(json_str)
   
   print(j)
   print(f"type j: {type(j)}")
   print(f"type item: {type(j['item'])}")
   
 
