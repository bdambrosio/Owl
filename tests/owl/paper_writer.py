import os, sys, logging, glob
import pandas as pd
import arxiv
from arxiv import Client, Search, SortCriterion, SortOrder
import ast
import concurrent
from csv import writer
#from IPython.display import display, Markdown, Latex
import json
import re
import random
import openai
import traceback
from lxml import etree
from PyPDF2 import PdfReader
import pdfminer.high_level as miner
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import requests
import urllib.request
import numpy as np
import faiss
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from termcolor import colored
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as lc
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec, QRect
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
from OwlCoT import LLM, ListDialog, generate_faiss_id
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from transformers import AutoTokenizer, AutoModel
import webbrowser
from Planner import Planner

# startup AI resources

# load embedding model and tokenizer
embedding_tokenizer = AutoTokenizer.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')
#load base model
#embedding_model = AutoModel.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')
#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
#embedding_model.load_adapter('/home/bruce/Downloads/models/Specter-2', source="hf", set_active=True)
#embedding_model.load_adapter('/home/bruce/Downloads/models/Specter-2')
from adapters import AutoAdapterModel

embedding_model = AutoAdapterModel.from_pretrained("allenai/specter2_aug2023refresh_base")
embedding_adapter_name = embedding_model.load_adapter("allenai/specter2_aug2023refresh", source="hf", set_active=True)

OPENAI_MODEL3 = "gpt-3.5-turbo-16k"
OPENAI_MODEL4 = "gpt-4-1106-preview"
#OPENAI_MODEL = "gpt-4-32k"
#EMBEDDING_MODEL = "text-embedding-ada-002"
ssKey = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
OS_MODEL='zephyr'

openai_api_key = os.getenv("OPENAI_API_KEY")
openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
memory = VolatileMemory()
#llm = LLM(None, memory, osClient=OSClient(api_key=None), openAIClient=openAIClient, template=OS_MODEL)
import OwlCoT as cot
cot = cot.OwlInnerVoice(None)
pl = Planner(None, cot)
import semanticScholar2 as s2
s2.cot = cot

class PWUI(QWidget):
    def __init__(self, rows, config_json):
        super().__init__()
        self.config = {}
        self.parent_config = config_json # Dictionary in calling_app space to store combo box references
        self.initUI(rows, config_json)

    def initUI(self, rows, result_json):
        layout = QVBoxLayout()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
        self.setPalette(palette)
        self.codec = QTextCodec.codecForName("UTF-8")
        self.widgetFont = QFont(); self.widgetFont.setPointSize(14)

        for row in rows:
            row_layout = QHBoxLayout()

            # Label for the row
            label = QLabel(row)
            row_layout.addWidget(label)

            # Yes/No ComboBox
            yes_no_combo_box = QComboBox()
            yes_no_combo_box.addItems(["Yes", "No"])
            row_layout.addWidget(yes_no_combo_box)

            # Model ComboBox
            model_combo_box = QComboBox()
            model_combo_box.addItems(["llm", "gpt3", "gpt4"])
            row_layout.addWidget(model_combo_box)

            # Store combo boxes in the dictionary with the row label as the key
            self.config[row] = (yes_no_combo_box, model_combo_box)

            layout.addLayout(row_layout)
        # Close Button
        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.on_close_clicked)
        layout.addWidget(close_button)
        self.setLayout(layout)

        self.setWindowTitle('Research Configuration')
        self.setGeometry(300, 300, 400, 250)
        self.show()

    def on_close_clicked(self):
        """
        Handles the close button click event. Retrieves values from the combo boxes
        and then closes the application.
        """
        for row_label, (yes_no_combo, model_combo) in self.config.items():
            yes_no_value = yes_no_combo.currentText()
            model_value = model_combo.currentText()
            self.parent_config[row_label] = {'exec':yes_no_value, 'model':model_value}
        self.close()

    def get_row_values(self, row_label):
        """
        Retrieve the values of the combo boxes for a given row label.
        Parameters:
        row_label (str): The label of the row.
        Returns:
        tuple: A tuple containing the selected values of the Yes/No and Model combo boxes.
        """
        if row_label in self.config:
            yes_no_value = self.config[row_label][0].currentText()
            model_value = self.config[row_label][1].currentText()
            return yes_no_value, model_value
        else:
            return None


import re

def extract_acronyms(text, pattern=r"\b[A-Za-z]+(?:-[A-Za-z\d]*)+\b"):
    """
    Extracts acronyms from the given text using the specified regular expression pattern.
    Parameters:
    text (str): The text from which to extract acronyms.
    pattern (str): The regular expression pattern to use for extraction.
    Returns:
    list: A list of extracted acronyms.
    """
    return re.findall(pattern, text)


def format_outline(json_data, indent=0):
    """
    Formats a research paper outline given in JSON into an indented list as a string.

    Parameters:
    json_data (dict): The research paper outline in JSON format.
    indent (int): The current level of indentation.

    Returns:
    str: A formatted string representing the research paper outline.
    """
    formatted_str = ""
    indent_str = " -" * indent

    if "title" in json_data:
        formatted_str += indent_str + ' '+json_data["title"] + "\n"

    if "sections" in json_data:
        for section in json_data["sections"]:
            formatted_str += format_outline(section, indent + 1)

    return formatted_str


entity_cache = {}
entity_cache_filepath = 'paper_writer_entity_cache.json'
if not os.path.exists(entity_cache_filepath):
    with open(entity_cache_filepath, 'w') as pf:
        json.dump(entity_cache, pf)
    print(f"created 'paper_entity_cache'")
else:
    try:
        with open(entity_cache_filepath, 'r') as pf:
            entity_cache=json.load(pf)
    except Exception as e:
        print(f'failure to load entity cache, repair or delete\n  {str(e)}')
        sys.exit(-1)
    print(f"loaded {entity_cache_filepath}")

def entities(paper_title, paper_outline, paper_summaries, ids):
    global entity_cache
    items = []
    total=len(ids); cached=0
    for id, excerpt in zip(ids, paper_summaries):
        # an 'excerpt' is [title, text]
        int_id = str(int(id)) # json 'dump' writes these ints as strings, so they won't match reloaded items unless we cast as strings
        if int_id in entity_cache:
            cached += 1
            items.extend(entity_cache[int_id])
        else:
            excerpt_items = extract_entities(paper_title, paper_outline, excerpt[0]+'\n'+excerpt[1])
            entity_cache[int_id]=excerpt_items
            items.extend(entity_cache[int_id])
    print(f'entities total {total}, in cache: {cached}')
    with open(entity_cache_filepath, 'w') as pf:
        json.dump(entity_cache, pf)
    print(f"wrote {entity_cache_filepath}")
    return list(set(items))

def extract_entities(paper_title, paper_outline, summary):
    kwd_messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
"""),
                  UserMessage(f"""Your current task is to extract all keywords and named-entities (which may appear as acronyms) important to the topic {paper_title} from the following research excerpt.

Respond using the following format:
{{"found": ["keywd1", "Named Entity1", "Acronym1", "keywd2", ...] }}

If distinction between keyword, acronym, or named_entity is unclear, it is acceptable to list a term or phrase under multiple categories.

<RESEARCH EXCERPT>
{summary}
</RESEARCH EXCERPT>

Remember, respond in JSON using the following format:
{{"found": ["keywd1", "Named Entity1", "Acronym1", "keywd2", ...]}}
"""),
                  AssistantMessage("")
              ]
    
    #response_json = cot.llm.ask('', kwd_messages, client=cot.llm.openAIClient, template='gpt-4-1106-preview', max_tokens=400, temp=0.1, validator=JSONResponseValidator())
    response_json = cot.llm.ask('', kwd_messages, client=cot.llm.osClient, max_tokens=400, temp=0.1, stop_on_json=True, validator=JSONResponseValidator())
    # remove all more common things
    keywords = []
    if 'found' in response_json:
        for word in response_json['found']:
            zipf = wf.zipf_frequency(word, 'en', wordlist='large')
            if zipf < 2.85 and word not in keywords:
                keywords.append(word)
    for word in extract_acronyms(summary):
        zipf = wf.zipf_frequency(word, 'en', wordlist='large')
        if zipf < 2.85 and word not in keywords:
            keywords.append(word)
    #print(f'\nKeywords: {keywords}\n')
    return keywords

def entities_to_str(item_list):
    return '\n'.join(item_list)
    
def rewrite(paper_outline, paper_title, section_outline, draft, query, paper_summaries, keywds, subsection_topic, subsection_token_length, parent_section_title, heading_1, heading_1_draft):
    missing_entities_prompt=f"""Your current task is to identify important missing entities in the DRAFT for the part titled: '{section_outline['title']}', to increase the density of relevant information it contains. 
The current DRAFT is:
<DRAFT>
{draft}
</DRAFT>

This DRAFT is the:
{section_outline["title"]}
within: {parent_section_title} 
{"within: "+heading_1 if heading_1 != parent_section_title else ''}
The draft should cover the specific topic: {subsection_topic}

A MISSING ENTITY is an ENTITY that is: 
 - In the ENTITIES list provided earlier;
 - Highly relevant and specific to the paper title: {paper_title}
 - Highly relevant to the current draft topic: {subsection_topic}; and role the content is to fill;
 - missing in the previous draft; 

Following these steps:
Step 1. Reason step by step to determine the role of this section within the paper. Do not include your reasoning in your response.
Step 2. Identify 3-5 MISSING ENTITIES to be added in the rewrite.
Step 3. Respond with the list of identified MISSING ENTITIES from step 2. List ONLY the selected missing entities, without any explanation or commentary. Your response should end with:

</MISSING_ENTITIES>
"""

    sysMessage = SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{format_outline(paper_outline)}

The following is a set of published research extracts used in creating the previous draft of this part of the paper"

<RESEARCH EXCERPTS>
{paper_summaries}
</RESEARCH EXCERPTS>

The following have been identified as key items mentioned in the above excerpts. These items may not have been included in the current draft:

<ENTITIES>
{entities_to_str(keywds)}
</ENTITIES>

The {heading_1} section content up to this point is:

<PRIOR_SECTION_CONTENT>
{heading_1_draft}
</PRIOR_SECTION_CONTENT>

"""
)
    short_sysMessage = SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{format_outline(paper_outline)}

The following have been identified as key Entities (key phrases, acronyms, or named-entities) for the paper.  These items may not have been included in the current draft:

<ENTITIES>
{entities_to_str(keywds)}
</ENTITIES>

The {heading_1} section content up to this point is:

<PRIOR_SECTION_CONTENT>
{heading_1_draft}
</PRIOR_SECTION_CONTENT>

"""
)

    entity_select_msgs = [short_sysMessage,
                          UserMessage(missing_entities_prompt),
                          AssistantMessage("<MISSING_ENTITIES>")
                          ]

    #response = cot.llm.ask('', entity_select_msgs, client=cot.llm.openAIClient, template='gpt-3.5-turbo-16k', max_tokens=100, temp=0.1, eos='</MISSING_ENTITIES>')
    #response = cot.llm.ask('', entity_select_msgs, client=cot.llm.openAIClient, template='gpt-4-1106-preview', max_tokens=100, temp=0.1, eos='</MISSING_ENTITIES>')
    response = cot.llm.ask('', entity_select_msgs, client=cot.llm.osClient, max_tokens=100, temp=0.1, eos='</MISSING_ENTITIES>')
    if response is None or len(response) == 0:
        return draft
    me = response
    end_idx = me.rfind('</MISSING_ENTITIES>')
    if end_idx < 0:
        end_idx = len(me)
        rewrite = me[:end_idx-1]
    print(f'\nMissing Entities:\n{me}\n')
    missing_entities = me

    rewrite_prompt=f"""Your current task is to rewrite the DRAFT for the part titled: '{section_outline['title']}', to increase the density of relevant information it contains. 
The current DRAFT is:

<DRAFT>
{draft}
</DRAFT>

This DRAFT is the:
{section_outline["title"]}
within: {parent_section_title} 
{"within: "+heading_1 if heading_1 != parent_section_title else ''}

Entities (key phrases, acronyms, or names-entities) that are missing in the above draft include:

<MISSING_ENTITIES>
{missing_entities}
</MISSING_ENTITIES>

The rewrite should cover the specific topic: {subsection_topic}
Following these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, denser draft of the same length which covers every Entity and detail from the previous draft as well as the MISSING_ENTITIES listed above.

Further Instructions: 
 - Coherence: re-write the previous draft to improve flow and make space for additional entities;
 - Missing entities should appear where best for the flow and coherence in the new draft;
 - Never drop entities from the previous content. If space cannot be made, add fewer new entities. 
 - Your goal is information density: use the same number of words as the previous draft, or as few more as needed. Remove or rewrite low-content phrases or sentences to improve conciseness.
 - Ensure the rewrite provides all the depth and information from the previous draft. 
 - Your response include ONLY the rewritten draft, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>
"""
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    #response = cot.llm.ask('', messages, client=cot.llm.openAIClient, template='gpt-3.5-turbo-16k', max_tokens=int(1.5*subsection_token_length), temp=0.1, eos='</DRAFT>')
    response = cot.llm.ask('', messages, client=cot.llm.osClient, max_tokens=int(1.5*subsection_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    print(f'\nRewrite:\n{rewrite}\n')
    return rewrite

plans_filepath = "./arxiv/arxiv_plans.json"
plans = {}

if os.path.exists(plans_filepath):
    with open(plans_filepath, 'r') as f:
        plans = json.load(f)
        print(f'loaded plans.json')
else:
    print(f'initializing plans.json')
    plans = {}
    with open(plans_filepath, 'w') as f:
        plans = json.dump(plans, f)

def plan_search():
    global plans
    plan = None
    if len(plans)> 0:
        plan_names = list(plans.keys())
        picker = ListDialog(plan_names)
        result = picker.exec()
        if result == QDialog.Accepted:
            selected_index = picker.selected_index()
            print(f'Selected Item Index: {selected_index}') 
            if selected_index != -1:  # -1 means no selection
                plan_name = plan_names[selected_index]
                plan = plans[plan_name]
                print(json.dumps(plan, indent=4))
                plans[plan['name']] = plan
                pl.active_plan = plan

    if plan is None:
        # new plan, didn't select any existing
        plan = pl.init_plan()
        plan = pl.analyze(plan)
        # store new plan in list of search plans
        plans[plan['name']] = plan
        with open(plans_filepath, 'w') as f:
            json.dump(plans,f)
        print(json.dumps(plan, indent=4))
        #   {"name":'plannnn_aaa', "dscp":'bbb', "sbar":{"needs":'q\na', "background":'q\na',"observations":'q\na'}

    if plan is None:
        return
    return plan

def s2_search (config, outline, title_prefix='', dscp_prefix=''):
    #
    ## call get_articles in semanticScholar2 to query semanticScholar for each section or subsection in article
    ## we can get 0 results if too specific, so probabilistically relax query as needed
    #
    if 'title' in outline:
        title_prefix += '. '+outline['title']
    if 'dscp' in outline:
        dscp_prefix += '. '+outline['dscp']
    if 'sections' in outline:
        for subsection in outline['sections']:
            s2_search(config, subsection, title_prefix, dscp_prefix)
    else:
        entities = extract_entities('', '', title_prefix+'\n'+dscp_prefix)
        # search entity list as query
        result_list = []
        query_list = entities.copy() # in case we expand later to try multiple descent paths
        while len(result_list) == 0 and len(query_list) > 0:
            result_list, total_papers, next_offset = s2.get_articles(' '.join(query_list), confirm=True)
            # if nothing, randomly remove one entity
            query_list.remove(random.choice(query_list))

def write_report(config, paper_outline=None, section_outline=None, length=400): 
    #rows = ["Query", "SBAR", "Outline", "WebSearch", "Write", "ReWrite"]
    query_config = config['Query']
    query = ""
    if query_config['exec'] == 'Yes':
        query = cot.confirmation_popup('Question to report on?', '')
        if not query:
            return None

    sbar_config = config['SBAR']
    if sbar_config['exec'] == 'Yes':
        plan = plan_search()
    else:
        plan = plan_search()
        
    outline_config = config['Outline']
    if outline_config['exec'] == 'Yes':
        # make outline
        pl.outline(config, plan)
        outline = plan['outline']
        cot.save_workingMemory() # save updated plan
    else:
        outline = default_outline # a canned research outline at bottom of this file.

    search_config = config['Search']
    # Note this is web search. Local faiss or other resource search will be done in Write below
    if search_config['exec'] == 'Yes':
        # do search - eventually we'll need subsearches: wiki, web, axriv, s2, self?, ...
        # also need to configure prompt, depth (depth in terms of # articles total or new per section?)
        s2_search(config, outline)

    write_config = config['Write']
    if write_config['exec'] == 'Yes':
        if section_outline is None:
            section_outline = paper_outline
        if 'length' in config:
            length = int(config['length'])
        else:
            length = 1200
        # write report! pbly should add length field, # rewrites
        write_report_aux(config, paper_outline=paper_outline, section_outline=section_outline, length=length)
        
    rewrite_config = config['ReWrite'] # not sure, is this just for rewrite of existing?
    if rewrite_config['exec'] == True:
        # rewrite draft, section by section, with prior critique input on what's wrong.
        pass


def write_report_aux(config, paper_outline=None, section_outline=None, length=400, inst='', topic='', paper_title='', abstract='', depth=0, parent_section_title='', parent_section_partial='', heading_1=None, heading_1_draft = '', num_rewrites=0):
    if len(paper_title) == 0 and depth == 0 and 'title' in paper_outline:
        paper_title=paper_outline['title']
    if 'length' in section_outline:
        length = section_outline['length']
    if 'rewrites' in section_outline:
        num_rewrites = section_outline['rewrites']
    if 'task' in section_outline:
        # overall instruction for this outline
        subsection_inst = inst+ '\n'+ section_outline['task']
    else:
        subsection_inst = inst
    subsection_topic = section_outline['dscp'] if 'dscp' in section_outline else section_outline['title']
    subsection_title = section_outline['title']
    #print(f"\nWRITE_PAPER section: {topic}")
    if 'sections' in section_outline:
        #
        ### write section intro first draft
        #
        subsection_depth = 1+depth
        num_sections = len(paper_outline['sections'])
        subsection_token_length = int(length/len(section_outline['sections']))
        section = ''
        n=0
        for subsection in section_outline['sections']:
            if depth == 0:
                heading_1 = subsection['title']
                heading_1_draft = ''
            print(f"subsection title {subsection['title']}")
            subsection_text = '\n\n'+'.'*depth+subsection['title']+'\n'+write_report_aux(config,
                                                                                    paper_outline=paper_outline,
                                                                                    section_outline=subsection,
                                                                                    length=subsection_token_length,
                                                                                    inst=subsection_inst,
                                                                                    topic=subsection_topic,
                                                                                    paper_title=paper_title,
                                                                                    abstract=abstract,
                                                                                    depth=subsection_depth,
                                                                                    parent_section_title=subsection_title,
                                                                                    parent_section_partial=section,
                                                                                    heading_1= heading_1,
                                                                                    heading_1_draft=heading_1_draft,
                                                                                    num_rewrites=num_rewrites)
            section += subsection_text
            heading_1_draft += subsection_text
            if depth==0:
                with open(f'section{n}.txt', 'w') as pf:
                    pf.write(section)
                n += 1
            
        return section
    
    elif 'title' in section_outline:
        # actuall llm call to write this terminal section
        section = section_outline['title']
        query = heading_1+', '+parent_section_title+' '+subsection_topic
        ids, excerpts = s2.search(query) # this assumes web searching has been donenote we aren't using dscp for search
        paper_summaries = '\n'.join(['Title: '+s[0]+'\n'+s[1] for s in excerpts])
        subsection_token_length = max(320,length) # no less than a paragraph
        
        #
        ### Write initial content
        #
        
        print(f"\nWriting:{section_outline['title']} length {length}\n  within {parent_section_title}\n     within{heading_1}\n covering {subsection_topic}")
        messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{format_outline(paper_outline)}

"""),
                  UserMessage(f"""Your current task is to write the part titled:
'{section_outline["title"]}'
within '{parent_section_title}' 
{"within "+heading_1 if heading_1 != parent_section_title else ''}
The following is a set of published research extracts for use in creating this section of the paper:

<RESEARCH EXCERPTS>
{paper_summaries}
</RESEARCH EXCERPTS>

Again, your current task is to write the part titled:
'{section_outline["title"]}'
within '{parent_section_title}' 
{"within "+heading_1 if heading_1 != parent_section_title else ''}
The {heading_1} section content up to this point is:

<PRIOR_SECTION_CONTENT>
{heading_1_draft}
</PRIOR_SECTION_CONTENT>

First reason step by step to determine the role of the part you are generating within the overall paper, 
then generate the appropriate text, subject to the following guidelines:

 - Output ONLY the text, do NOT output your reasoning.
 - Write a concise, detailed text using known fact and the above research excepts, of about {length} words in length.
 - This section should cover the specific topic: {parent_section_title} - {subsection_topic}
 - You may refer to, but not repeat, prior section content in any text you produce.
 - The section should be about {subsection_token_length} words long. The goal is to present an integrated view of the assigned topic, highlighting controversy where present, and capturing the overall state of knowledge along with essential statements, methods, observations, inferences, hypotheses, and conclusions. This must be done in light of the place of this section or subsection within the overall paper.

 - Please ensure the section provides depth while removing redundant or superflous detail, ensuring that every critical aspect of the source argument, observations, methods, findings, or conclusions is included in the list of key points.
End the section as follows:

</DRAFT>
"""),
              AssistantMessage("<DRAFT>\n")
              ]
        response = cot.llm.ask('', messages, client=cot.llm.osClient, max_tokens=subsection_token_length, temp=0.1, eos='</DRAFT>')
        #response = cot.llm.ask('', messages, client=cot.llm.openAIClient, template='gpt-4-1106-preview', max_tokens=subsection_token_length, temp=0.1, eos='</DRAFT>')
        end_idx = response.rfind('</DRAFT>')
        if end_idx < 0:
            end_idx = len(response)
        draft = response[:end_idx]

        print(f'\nFirst Draft:\n{draft}\n')
        if num_rewrites < 1:
            return draft

        #
        ### Now do rewrites
        #
        ### first collect entities
        #
        keywds = entities(paper_title, paper_outline, excerpts, ids)

        for i in range(num_rewrites):
            draft = rewrite(paper_outline, paper_title, section_outline, draft, query, paper_summaries, keywds, subsection_topic, 2*subsection_token_length, parent_section_title, heading_1, heading_1_draft)
    return draft


if __name__ == '__main__':
    default_outline = {
        "task":
        """write a detailed research survey on circulating miRNA and DNA methylation patterns as biomarkers for early stage lung cancer detection.""",
        "title":"Circulating miRNA and DNA methylation patterns as biomarkers for early stage lung cancer detection.",
        "sections":[
            {"title":"Introduction", "length": 600, "rewrites":1,
             "sections":[
                 {"title":"Overview of Lung Cancer",
                  "dscp":"Prevalence and impact and importance of early detection for improved prognosis"},
                 {"title":"Biomarkers in Cancer Detection", "rewrites": 2,
                  "dscp":"Definition and role of biomarkers in cancer. Traditional biomarkers used in lung cancer"},
                 {"title":"Emerging Biomarkers: miRNA and DNA Methylation", "rewrites":2,
                  "dscp":"Explanation of miRNA and DNA methylation. Potential advantages over traditional biomarkers"},
             ]
             },
                 
            {"title":"Circulating miRNA as Biomarkers", "length":1000,"rewrites":2,
             "sections":[
                 {"title":"Biological Role of miRNAs",
                  "sections":[
                      {"title":"Function and significance in cell regulation"},
                      {"title":"How alterations in miRNA can contribute to cancer development"},
                  ],
                  },
                 {"title":"miRNAs in Lung Cancer",
                  "sections":[
                      {"title":"Studies showing miRNA expression profiles in lung cancer patients"},
                      {"title":"Specific miRNAs frequently dysregulated in lung cancer", "length":300},
                  ],
                  },
                 {"title":"Circulating miRNAs",
                  "dscp":"Circulating miRNAs in blood and other bodily fluids. Advantages as non-invasive biomarkers"},
                 {"title":"Studies on miRNAs in Early Lung Cancer Detection",
                  "dscp":"Summary of key research findings. Analysis of sensitivity, specificity, and overall effectiveness"}
             ]
             },
            
            {"title":"DNA Methylation Patterns as Biomarkers", "length":1000,"rewrites":2,
             "sections":[
                 {"title":"Basics of DNA Methylation",
                  "sections":[
                      {"title":"Role in gene expression and regulation"},
                      {"title":"How aberrant methylation patterns are linked to cancer"},
                  ],
                  },
                 {"title":"DNA Methylation in Lung Cancer",
                  "sections":[
                      {"title":"Commonly observed methylation changes in lung cancer"},
                      {"title":"Genes frequently affected by methylation in lung cancer"},
                  ],
                  },
                 {"title":"Research on DNA Methylation Patterns in Early Lung Cancer",
                  "sections":[
                      {"title":"Overview of significant studies"},
                      {"title":"Discussion on the feasibility and accuracy"},
                  ],
                  },
                 {"title":"Detection of Methylation Patterns",
                  "sections":[
                      {"title":"Techniques used to detect DNA methylation"},
                      {"title":"Challenges in using methylation patterns for early detection"},
                  ],
                  },
             ]
             },
            
            {"title":"Comparative Analysis: miRNA and DNA Methylation Biomarkers", "rewrites":2,
             "dscp":"miRNA and DNA Methylation Biomarkers",
             "sections":[
                 {"title":"Comparison in terms of reliability, cost, and accessibility"},
                 {"title":"miRNA vs. DNA Methylation Biomarkers - Potential for combining both markers for improved detection",
                  "length":1000, "rewrites":3},
             ],
             },
            
            {"title":"Near-term Opportunities, Challenges and Future directions", "length":1200, "rewrites":2,
             "sections":[
                 {"title":"Near-term Opportunites for combined miRNA / DNA methylation as biomarkers", "length":800,
                  "dscp":"near-term opportunities in utilizing these biomarkers for blood (cell free, tumor-cell, and vesicle) assay of early lung cancer",
                  },
                 {"title":"Challenges",
                  "dscp":"Technical and clinical challenges in utilizing these biomarkers for cell-free serum assay for early cancer detection",
                  "sections":[
                      {"title":"Challenges in utilizing miRNA and DNA methylation as biomarkers"},
                      {"title":"Standardization and validation"},
                  ]
                  },
                 {"title":"Future Research",
                  "dscp": "areas needing further development for using circulating miRNA and DNA methylation assay in early stage cancer detection",
                  "sections":[
                      {"title":"Areas needed further investigation"},
                      {"title":"Potential advances in technology and methodology"}
                  ]
                  },
             ]
             },
            
            {"title":"Conclusion", "rewrites":2,
             "sections":[
                 {"title":"Summary of Current Understanding",
                  "dscp":"Recap of the potential of circulating miRNA and DNA methylation patterns in early lung cancer detection",
                  },
                 {"title":"Prediction",
                  "dscp": "Prediction for when assay of circulating miRNA and DNA methylation will appear in clinical practise"},
             ]
             }
        ]
    }
    
    #print(format_outline(outline))
    config = {}
    app = QApplication(sys.argv)
    rows = ["Query", "SBAR", "Outline", "Search", "Write", "ReWrite"]
    ex = PWUI(rows, config)

    # Example usage to get combo box values for "Row 2"

    app.exec_()
    print(config)
    try:
        cot.display_response('calling paper_writer')
        paper = write_report(config, paper_outline=default_outline, section_outline=default_outline)
    except Exception as e:
        traceback.print_exc()
        print(str(e))
        sys.exit(-1)
    #print(f'\n\n\n\n{paper}')
    #with open('paper.txt', 'w') as pf:
    #    pf.write(paper)
    #print(json.dumps(outline, indent=2))
    
