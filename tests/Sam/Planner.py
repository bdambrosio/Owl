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
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial
from SamCoT import ListDialog, LLM

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
    {"action": "none", "arguments": "None", "result": "None", "description": "no action is needed."},
    {"action": "append", "arguments": "name1, name2", "result": "name3", "description": "append the Item name1 to List name2, and assign the resulting list to name3"},
    {"action": "article", "arguments": "name1", "result": "name2", "description": "access the article title named name1, use it to retrieve the article body, and assign it to name2."},
    {"action": "assign", "arguments": "literal1", "result": "name1", "description": "create a new Item with value literal1 and assign it to name1"},
    {"action": "choose", "arguments": "name1, "name2", "result": "name3", "description": "choose an item from the list name1, according to the criteria in name2, and assign it to name3"},
    {"action": "concatenate", "arguments": "name1, name2", "result": "name3", "description": "append the list named list2 to the list named list1 and assign the resulting list to name3"},
    {"action": "difference", "arguments": "item1, item2", "result": "item3", "description": "identify content in item1 and not in item2 and assign it to name item3"},
    {"action": "empty", "arguments": "list1", "result": "item1", "description": "test if list1 is empty and assign the boolean True/False accordingly to item1."},
    {"action": "extract", "arguments": "item1, item2", "result": "item3", "description": "extract content related to item1 from item2 and assign it to item3"},
    {"action": "first", "arguments": "list1", "result": "item1", "description": "select the first item in list1 and assign it to item1."},
    {"action": "gpt4", "arguments": "item1", "result": "item2", "description": "ask gpt4 item1 and assign the response to item2"},
    {"action": "integrate", "arguments": "item1 ,item2", "result": "item3", "description": "combine item1 and item2 into a single coherent item and assign it to item3."},
    {"action": "question", "arguments": "name1", "result": "name2", "description": "access the item name1, present it to doc, and assign his response to name2."},
    {"action": "recall", "arguments": "name1", "result": "item1", "description": "retrieve item named item1 from working memory and assign it to name item1."},
    {"action": "request", "arguments": "item1", "result": "item2", "description": "request a specific web resource with url item1 and assign the result to name item2."},
    {"action": "sort", "arguments": "list1, item1", "result": "list2", "description": "rank the items in list1 by criteria item1 and assign the sorted list to name list2. Returns a list in ranked order, best first."},
    {"action": "tell", "arguments": "item1", "result": "None", "description": "present item1 to the user."},
    {"action": "web", "arguments": "item1", "result": "item2", "description": "perform a web search, using the item1 as the search string, and assign the resulting  integrated content to name item2."},
    {"action": "wiki", "arguments": "item1", "result": "item2", "description": "wiki search the local wikipedia database using item1 as the search string, and assign the integrated content to item2."}
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
Your plan should be structured as a list of instantiated actions from the above action list. Each action instantiation will include the specified action, the arguments for that action, and the name to assign to the result. 
All names must appear in a result before the can be referenced as arguments.
The order of the actions in the plan matters, as it represents the sequence in which they should be executed.

Example Plan (1-shot):

Plan:
[
{"action": "assign", "arguments": "'Apple', item1", "result": "item1"},
{"action": "assign", "arguments": EMPTYLIST, "result": "list1"},
{"action": "assign", "arguments": "5", "result": "number1"},
{"action": "append", "arguments": "number1, myList", "result": "updatedList"},
{"action": "gpt4", "arguments": "What is the meaning of life?", "result": "response1"},
{"action": "tell", "arguments": "response1", "result": "None"}
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


   
    def eval_AWM (self):
       #
       ## note this and samCoT should be written in a way that doesn't require direct access/manipulation of samCoT data!
       #
       names=[f"{self.samCoT.active_WM[item]['name']}: {str(self.samCoT.active_WM[item]['item'])[:32]}" for item in self.samCoT.active_WM]
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
          entry = self.samCoT.active_WM[name]
          print(f'selected action: {entry}')
          if type(entry['item']) != dict:
             try:
                print(f"trying item string as json {entry['item']}")
                dict_item = json.loads(entry['item'].strip())
             except Exception as e:
                print(f"json loads failed, trying (gasp!) eval")
                dict_item = eval(entry['item'].strip())
                if type(dict_item) != dict:
                   self.ui.display_response(f'invalid json {str(e)}')
                   continue
                
          valid_json=True
          # ok, item is a dict, let's see if it is an action
          dict_item = entry['item']
          if 'action' not in dict_item:
             self.ui.display_response(f'item is not an action {item}')
             continue
          elif dict_item['action'] == 'first':
             return self.first(dict_item)
          elif dict_item['action'] == 'assign':
             return self.assign(dict_item)
          elif dict_item['action'] == 'web':
             return self.assign(dict_item)
          elif dict_item['action'] == 'extract':
             return self.pweb(dict_item)
          elif dict_item['action'] == 'tell':
             return self.ptell(dict_item)
          elif dict_item['action'] == 'wiki':
             return self.pwiki(dict_item)
          return 'no item selected or no item found'
       return 'action not yet implemented'
   
    def LLM_one_op(self, operation_prompt, data1=None, data2=None, validator = DefaultResponseValidator()):
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


    def parse_as_action(self, item):
       if type(item) is not dict or 'action' not in item or 'arguments' not in item or 'result' not in item:
          self.ui.display_response(f'form is not an action {item}')
          raise InvalidAction(str(item))
       else:
          return item['action'], item['arguments'], item['result']


    def assign(self, action):
       #
       ## assign an item or literal as the value of a name
       ## example: {"action":"assign", "arguments":"abc", "result":"pi35"}
       ##   assigns the literal string 'abc' as the value of active memory name pi35 
       ##   if pi35 is not present as a name in active memory, it is created 
       ##   should we recreate key? Not clear, as use case and semantics of key is unclear.
       ##   assume for now assign will be used for simple forms that will be referred to primarily by name, not key.
       print(f'assign {action}')
       action, arguments, result = self.parse_as_action(action)
       if type(result) != str: # target of assign must be a name 
          raise InvalidAction(f'target of assign must be a name: {str(item)}')       
       self.samCoT.create_AWM(arguments, name=result, confirm=False)

    def article(self, titleAddr):
       pass
    
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
          SystemMessage('The following is a list. Please select and respond with only the first entry. Include the entire first entry in your response.'),
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
You've always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.
Your conversation style is warm, gentle, humble, and engaging. """
       self.interpreter = PlanInterpreter(self.ui, self.samCoT, self, model=self.model)


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
    
    def analyze(self, prefix, form):
       self.sbar = None
       if prefix is None:
          prefix = 'tempClassName'
          class_prefix_prompt = [SystemMessage(f"""Return a short camelCase name for a python class supporting the following task. Respond in JSON using format: {{"name": '<pythonClassName>'}}.\nTask:\n{form}""")]
          prefix_json = self.llm.ask(form, class_prefix_prompt, max_tokens=100, temp=0.01, validator=JSONResponseValidator())
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
             user_prompt = self.llm.ask('', messages, temp = 0.05)
             print(f"\nAI : {step}, {user_prompt}")
             past = ''
             if self.sbar is not None and type(self.sbar) is dict and step in self.sbar.keys():
                past = self.sbar[step] # prime user response with last time
             readline.set_startup_hook(lambda: readline.insert_text(past))
             try:
                user_input = input(user_prompt)
             finally:
                readline.set_startup_hook()
             user_responses[step] = user_prompt+'\n'+user_input
             messages.append(AssistantMessage(user_prompt))
             messages.append(UserMessage(user_input))
          else: # closing AI thoughts and user feedback. No need to add to messages because no more iterations
             observations = self.llm.ask('', messages, max_tokens=150,temp = 0.05)
             user_responses['observations']=observations
             print(f"\nAI : {step}, {observations}")
             user_response = input("User: ")
             user_responses['observations']=observations+'\n'+user_response
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
       return f"\nTASK:\n{self.sbar['needs']}\nBackground:\n{self.sbar['background']}\nReview:\n{self.sbar['observations']}\nEND TASK\n"


    def plan(self):
    
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
       print(f'******* Developing Plan for {self.prefix}')

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
          messages = [SystemMessage("""You're tasked with creating a plan using a set of predefined actions. Each action has specific arguments and results. Below is the structure you should follow:\n'+planner_nl_list_prompt"""),
                      SystemMessage(f'\nYou have the following actions available:\n{action_primitive_descriptions}\n'),
                      SystemMessage(plan_prompt),
                      UserMessage(f'TaskName: {self.prefix}\n{self.sbar_as_text()}')
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
          plan = self.llm.ask('', messages, template='gpt-4',max_tokens=2000, temp=0.1, validator=DefaultResponseValidator())
          #plan, plan_steps = ut.get_plan(plan_text)
          print(f'***** Plan *****\n{plan}\n\nPlease review and critique or <Enter> when satisfied')
          
          user_critique = input('Critique: ')
          if len(user_critique) <4:
             user_satisfied = True
             print("*******user satisfied with plan")
             break
          else:
             print('***** user not satisfield, retrying')
         
       self.save_plan(self.prefix, plan)
       # experiment - did we get an understandable plan?
       step = self.interpreter.first(plan)
       print(step)
       return plan     

if __name__ == '__main__':
   import Sam as sam
   ui = sam.window
   cot = ui.samCoT
   pl = Planner(ui, cot)
   print('******analyze news**************************')
   pl.analyze('news',"build a short list of fact-checked world news sources on the web")
   print('******plan news********************')
   plan = pl.plan()
   print(f'\nplan\n{plan}\n')
   print('******extract first step from news plan ***')
   step = pl.interpreter.first(plan)
   print('*******test if first step is executable *************')
   pl.interpreter.test_executable(step)
