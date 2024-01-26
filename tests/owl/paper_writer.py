import os, sys, logging, glob, time
import pandas as pd
import arxiv
from arxiv import Client, Search, SortCriterion, SortOrder
import ast
import concurrent, subprocess
from csv import writer
#from IPython.display import display, Markdown, Latex
import json
import re
import random
import argparse
import pickle
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
from PyQt5.QtCore import pyqtSignal
import signal
from OwlCoT import LLM, ListDialog, generate_faiss_id,OPENAI_MODEL3,OPENAI_MODEL4
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from transformers import AutoTokenizer, AutoModel
import webbrowser
import rewrite as rw
import jsonEditWidget as ew

# startup AI resources

# load embedding model and tokenizer
embedding_tokenizer = AutoTokenizer.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')
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

import OwlCoT as oiv
cot = oiv.OwlInnerVoice(None)
print (type(cot), type(cot.llm))
print (cot.template)
# set cot for rewrite so it can access llm
rw.cot = cot

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
            model_combo_box.addItems(["llm", "gpt3", "gpt4", "mistral-small", "mistral-medium"])
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

def get_template(row, config):
    global cot
    if row in config:
        if 'model' in config[row]:
            model = config[row]['model']
            if model == 'gpt3':
                return OPENAI_MODEL3
            if model == 'gpt4':
                return OPENAI_MODEL4
            if model == 'mistral-small':
                return 'mistral-small'
            if model == 'mistral-medium':
                return 'mistral-medium'
            if model == 'llm':
                return cot.template
        else:
            return cot.template
    print(f'get_template fail {row}\n{json.dumps(config, indent=2)}')
        
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

def literal_missing_entities(entities, draft):
    # identifies items in the entities_list that DO appear in the summaries but do NOT appear in the current draft.
    missing_entities = entities.copy()
    draft_l = draft.lower()
    for entity in entities:
        if entity.lower() in draft_l :
            missing_entities.remove(entity)
    #print(f'Missing entities from draft: {len(missing_entities)}')
    return missing_entities

def literal_included_entities(entities, text):
    # identifies items in the entities_list that DO appear in the current draft.
    included_entities = []
    text_l = text.lower()
    for entity in entities:
        if entity.lower() in text_l:
            included_entities.append(entity)
    #print(f'Missing entities from draft: {missing_entities}')
    return included_entities

def entities(paper_title, paper_outline, paper_summaries, ids, template):
    global entity_cache
    print(ids)
    items = []
    total=len(ids); cached=0
    for id, excerpt in zip(ids, paper_summaries):
        # an 'excerpt' is [title, text]
        int_id = str(int(id)) # json 'dump' writes these ints as strings, so they won't match reloaded items unless we cast as strings
        if int_id in entity_cache:
            cached += 1
            items.extend(entity_cache[int_id])
        else:
            excerpt_items = rw.extract_entities(excerpt[0]+'\n'+excerpt[1], title=paper_title, outline=paper_outline, template=template)
            entity_cache[int_id]=excerpt_items
            items.extend(entity_cache[int_id])
    print(f'entities total {total}, in cache: {cached}')
    with open(entity_cache_filepath, 'w') as pf:
        json.dump(entity_cache, pf)
    print(f"wrote {entity_cache_filepath}")
    return list(set(items))

def count_keyphrase_occurrences(texts, keyphrases):
    """
    Count the number of keyphrase occurrences in each text.

    Parameters:
    texts (list): A list of texts.
    keyphrases (list): A list of keyphrases to search for in the texts.

    Returns:
    list: A list of tuples, each containing a text and its corresponding keyphrase occurrence count.
    """
    counts = []
    for text in texts:
        count = sum(text.count(keyphrase.strip()) for keyphrase in keyphrases)
        counts.append((text, count))
    
    return counts

def select_top_n_texts(texts, keyphrases, n):
    """
    Select the top n texts with the maximum occurrences of keyphrases.

    Parameters:
    texts (list): A list of texts.
    keyphrases (list): A list of keyphrases.
    n (int): The number of texts to select.

    Returns:
    list: The top n texts with the most keyphrase occurrences.
    """
    #print(f'select top n texts type {type(texts)}, len {len(texts[0])}, keys {keyphrases}')
    #print(f'select top n texts {keyphrases}')
    counts = count_keyphrase_occurrences(texts, keyphrases)
    # Sort the texts by their counts in descending order and select the top n
    sorted_texts = sorted(counts, key=lambda x: x[1], reverse=True)
    return sorted_texts[:n]



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

def save_plans():
    global plans
    with open(plans_filepath, 'w') as f:
        json.dump(plans, f)
    
def plan_search():
    # note this uses arxiv/arxiv_plans.json, NOT working memory! why? no idea.
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

def make_search_queries(outline, section_outline, sbar, template):
    prompt = """
Following is an outline for a research paper, in JSON format: 

{{$outline}}

From this outline generate 3 SemanticScholar search queries for the section:
{{$target_section}}

A query can contain no more than 100 characters, Respond in a plain JSON format without any Markdown or code block formatting,  using the following format:
{"query1":'query 1 text',"query2": 'query 2 text', "query3": query 3 text'}

Respond ONLY with the JSON above, do not include any commentary or explanatory text.
"""
    messages = [SystemMessage(prompt),
                AssistantMessage('')
                ]
    queries = cot.llm.ask({"outline":json.dumps(outline, indent=2), "target_section":json.dumps(section_outline, indent=2)}, messages, stop_on_json=True, template=template, max_tokens=150, validator=JSONResponseValidator())
    if type(queries) is dict:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{json.dumps(queries, indent=2)}')
    else:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{queries}')
    return queries
    
def s2_search (config, outline, section_outline, sbar=None):
    #
    ### Note - ALL THIS IS DOING IS PRE-LOADING LOCAL LIBRARY!!! Doesn't need to return anything!
    ## call get_articles in semanticScholar2 to query semanticScholar for each section or subsection in article
    ## we can get 0 results if too specific, so probabilistically relax query as needed
    #
    #print(f'paper_writer entered s2_search with {section_outline}')
    template = get_template('Search', config)
    #print(f' paper_writer s2_search template: {template}')
    # llms sometimes add an empty 'sections' key on leaf sections.
    if 'sections' in section_outline and len(section_outline['sections']) > 0: 
        for subsection in section_outline['sections']:
            s2_search(config, outline, subsection, sbar)
    else:
        queries = make_search_queries(outline, section_outline, sbar, template)
        bads = ['(',')',"'",' AND', ' OR'] #tulu sometimes tries to make fancy queries, S2 doesn't like them
        if type(queries) is not dict:
            print(f's2_search query construction failure')
            return None
        for i in range(3):
            if 'query'+str(i) not in queries:
                continue
            query = queries['query'+str(i)]
            for bad in bads:
                query = query.replace(bad, '')
                result_list, total_papers, next_offset = s2.get_articles(query, confirm=True)
                print(f's2_search found {len(result_list)} new papers')
                while len(result_list) == 0 and next_offset < total_papers\
                      and cot.confirmation_popup('Continue?', ''):
                    s2.get_articles(query, next_offset, confirm=True)
                
# global passed into jsonEditor widget
updated_json = None

def handle_json_editor_closed(result):
    global updated_json
    print(f'got result {result}')
    updated_json = result

config = {}

def write_report(app, topic):
    global updated_json, config
    #rows = ["Query", "SBAR", "Outline", "WebSearch", "Write", "ReWrite"]
    plan = plan_search()
    if 'config' in plan:
        config = plan['config']
    rows = ["Query", "SBAR", "Outline", "Search", "Write", "ReWrite"]
    ex = PWUI(rows, config) # get config for this report task
    ex.show()
    app.exec()
    #get updated config, in case it changed
    if config is not None and len(config.keys()) > 0:
        plan['config'] = config
        save_plans()

    query_config = config['Query']
    query = ""
    if query_config['exec'] == 'Yes':
        query = cot.confirmation_popup('Question to report on?', '')
        save_plans()
    elif 'task' not in plan.keys():
        plan['task']=query
            
    sbar_config = config['SBAR']
    if sbar_config['exec'] == 'Yes':
        # pass in query to sbar!
        print(f'sbar input plan\n{plan}')
        if 'sbar' in plan and type(plan['sbar']) is dict and cot.confirmation_popup('Edit existing sbar?', json.dumps(plan['sbar'], indent=2)):
            # we already have an sbar, edit it
            app = QApplication(sys.argv)
            editor = ew.JsonEditor(plan['sbar'])
            editor.closed.connect(handle_json_editor_closed)
            editor.show()
            app.exec()
            print(f'SBAR: {updated_json}')
            if updated_json is not None:
                plan['sbar'] = updated_json
                save_plans()
        else:
            plan = pl.analyze(plan)
            save_plans()
        
    outline_config = config['Outline']
    if outline_config['exec'] == 'Yes':
        if 'outline' in plan and type(plan['outline']) is dict\
           and cot.confirmation_popup('Edit existing outline?', json.dumps(plan['outline'], indent=2)) is not None:
            # we already have an outline, edit it
            app = QApplication(sys.argv)
            editor = ew.JsonEditor(plan['outline'])
            editor.closed.connect(handle_json_editor_closed)
            editor.show()
            app.exec()
            #print(f'outline: {updated_json}')
            if updated_json is not None:
                plan['outline'] = updated_json
        else:
            # make outline
            plan = pl.outline(outline_config, plan)
            outline = plan['outline']
        save_plans() # save paper_writer plan memory
    else:
        outline = plan['outline']
        
    search_config = config['Search']
    # Note this is web search. Local faiss or other resource search will be done in Write below
    if search_config['exec'] == 'Yes':
        # do search - eventually we'll need subsearches: wiki, web, axriv, s2, self?, ...
        # also need to configure prompt, depth (depth in terms of # articles total or new per section?)
        s2_search(config, outline, outline)

    write_config = config['Write']
    if write_config['exec'] == 'Yes':
        if 'length' in config:
            length = int(config['length'])
        else:
            length = 1200
    # write report! pbly should add # rewrites
    write_report_aux(config, paper_outline=outline, section_outline=outline, length=length)
        
    rewrite_config = config['ReWrite'] # not sure, is this just for rewrite of existing?
    if rewrite_config['exec'] == True:
        # rewrite draft, section by section, with prior critique input on what's wrong.
        pass


def write_report_aux(config, paper_outline=None, section_outline=None, excerpts=None, length=400, dscp='', topic='', paper_title='', abstract='', depth=0, parent_section_title='', parent_section_partial='', heading_1_title='', heading_1_draft = '', num_rewrites=1, resources=None):
    template = get_template('Write',config)
    if depth == 0: #set section number initially to 0
        n = 0; refs=[]
    if len(paper_title) == 0 and depth == 0 and 'title' in paper_outline:
        paper_title=paper_outline['title']
    if 'length' in section_outline:
        length = section_outline['length']
    if 'rewrites' in section_outline:
        num_rewrites = section_outline['rewrites']

    # subsection dscp is full path dscp descending from root
    subsection_dscp = dscp
    if 'task' in section_outline:
        # overall instruction for this subsection
        subsection_dscp += '\n'+ section_outline['task']
        
    # subsection topic is local title or dscp 
    subsection_topic = section_outline['dscp'] if 'dscp' in section_outline else section_outline['title']
    subsection_title = section_outline['title']
    #print(f"\nWRITE_PAPER section: {topic}")
    if 'sections' in section_outline and len(section_outline['sections']) > 0:
        #
        ### write section intro first draft
        #
        subsection_depth = 1+depth
        num_sections = len(section_outline['sections'])
        subsection_token_length = int(length/len(section_outline['sections']))
        section = ''
        n=0
        for subsection in section_outline['sections']:
            if depth == 0:
                heading_1_title = subsection['title']
                heading_1_draft = ''
            print(f"subsection title {subsection['title']}")
            text, subsection_refs =\
                write_report_aux(config,
                                 paper_outline=paper_outline,
                                 section_outline=subsection,
                                 excerpts=excerpts,
                                 length=subsection_token_length,
                                 dscp=subsection_dscp,
                                 topic=subsection_topic,
                                 paper_title=paper_title,
                                 abstract=abstract,
                                 depth=subsection_depth,
                                 parent_section_title=subsection_title,
                                 parent_section_partial=section,
                                 heading_1_title= heading_1_title,
                                 heading_1_draft=heading_1_draft,
                                 num_rewrites=num_rewrites,
                                 resources=resources)
            subsection_text = '\n\n'+'.'*depth+subsection['title']+'\n'+text
            section += subsection_text
            for ref in subsection_refs:
                if ref not in refs:
                    refs.append(ref)
            heading_1_draft += subsection_text
            if depth==0:
                with open(f'section{n}.txt', 'w') as pf:
                    pf.write(section +'\n\nReferences:+\n'+'\n'.join(refs))
                n += 1
            
        if depth != 0:
            return section, refs
        else:
            return section +'\n\nReferences:+\n'+'\n'.join(refs)
    
    else:
        # no subsections, write this terminal section
        section = '' if 'sections' not in section_outline else section_outline['title']
        print(f'heading_1 {heading_1_title}\npst {parent_section_title}\nsubsection topic {subsection_topic}')
        query = heading_1_title+', '+parent_section_title+' '+subsection_topic
        # below assumes web searching has been done
        if resources is None:
            # do local search, excerpts to use not provided
            ids, excerpts = s2.search(query, subsection_dscp) 
        else:
            ids = resources['sections'][0]
            excerpts = resources['sections'][1]
        paper_summaries = '\n'.join(['Title: '+s[0]+'\n'+s[1] for s in excerpts])
        subsection_refs =  []
        for ref in [s[0]for s in excerpts]:
            if ref not in subsection_refs:
                subsection_refs.append(ref)
        subsection_token_length = max(500,length) # no less than a paragraph
        
        #
        ### Write initial content
        #
        
        print(f"\nWriting:{section_outline['title']} length {length}\n covering {subsection_topic}")
        messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{format_outline(paper_outline)}

"""),
                  UserMessage(f"""Your current task is to write the part titled: '{section_outline["title"]}'

<RESEARCH EXCERPTS>
{paper_summaries}
</RESEARCH EXCERPTS>


The {heading_1_title} section content up to this point is:

<{heading_1_title}_SECTION_CONTENT>
{heading_1_draft}
</{heading_1_title}_SECTION_CONTENT>

Again, your current task is to write the next part, titled: '{section_outline["title"]}'

1. First reason step by step to determine the role of the part you are generating within the overall paper, 
2. Then generate the appropriate text, subject to the following guidelines:

 - Output ONLY the text, do NOT output your reasoning.
 - Write a dense, detailed text using known fact and the above research excepts, of about {subsection_token_length} words in length.
 - This section should cover the specific topic: {parent_section_title}: {subsection_topic}
 - You may refer to, but not repeat, prior section content in any text you produce.
 - Present an integrated view of the assigned topic, noting controversy where present, and capturing the overall state of knowledge along with essential statements, methods, observations, inferences, hypotheses, and conclusions. This must be done in light of the place of this section or subsection within the overall paper.
 - Ensure the section provides depth while removing redundant or superflous detail, ensuring that every critical aspect of the source argument, observations, methods, findings, or conclusions is included.
End the section as follows:

</DRAFT>
"""),
              AssistantMessage("<DRAFT>\n")
              ]
        response = cot.llm.ask('', messages, template=template, max_tokens=int(subsection_token_length), temp=0.1, eos='</DRAFT>')
        end_idx = response.rfind('</DRAFT>')
        if end_idx < 0:
            end_idx = len(response)
        draft = response[:end_idx]

        #print(f'\nFirst Draft:\n{draft}\n')
        if num_rewrites < 1:
            return draft, subsection_refs

        #
        ### Now do rewrites
        #
        ### first collect entities
        #
        keywds = entities(paper_title, paper_outline, excerpts, ids, template)
        missing_entities = literal_missing_entities(keywds, draft)
        #print(f'\n missing entities in initial draft {len(missing_entities)}\n')
        for i in range(num_rewrites):
            if i < num_rewrites-1:
                template = template
            else:
                template = get_template('ReWrite', config)
            if i < num_rewrites-1:
                #add new entities
                draft = rw.add_pp_rewrite(paper_title, section_outline['title'], draft, paper_summaries, keywds, subsection_topic, int((1.3**(i+1))*subsection_token_length), parent_section_title, heading_1_title, heading_1_draft, template)
            else:
                # refine in final rewrite
                draft = rw.rewrite(paper_title, section_outline['title'], draft, paper_summaries, keywds, subsection_topic, 2*subsection_token_length, parent_section_title, heading_1_title, heading_1_draft, template)
            missing_entities = literal_missing_entities(keywds, draft)
            print(f'\n missing entities after rewrite {len(missing_entities)} \n')

        
        section = draft
        # make sure we write out top level sections even if they have no subsections!
        if depth==0: # single top-level section with no subsections
            with open(f'section{n}.txt', 'w') as pf:
                pf.write(draft)
            n += 1

    if depth == 0:
        print(f'\nSection:\n{section}\n')
        print(f'Refs:\n{subsection_refs}\n')
        
    return section, subsection_refs

class DisplayApp(QtWidgets.QWidget):
    def __init__(self, query, sections, dscp, template):
        super().__init__()
        self.query = query
        self.sections = sections
        self.dscp = dscp
        self.template = template

        self.memory_display = None
        self.windowCloseEvent = self.closeEvent
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
        self.setPalette(palette)
        self.codec = QTextCodec.codecForName("UTF-8")
        self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
        
        # Main Layout
        main_layout = QHBoxLayout()
        # Text Area
        text_layout = QVBoxLayout()
        main_layout.addLayout(text_layout)
        
        class MyTextEdit(QTextEdit):
            def __init__(self, app):
                super().__init__()
                self.app = app
                self.textChanged.connect(self.on_text_changed)
                
            def on_text_changed(self):
                #legacy from Owl, but who knows
                pass
            
            def keyPressEvent(self, event):
                #legacy from Owl, but who knows
                if event.matches(QKeySequence.Paste):
                    clipboard = QApplication.clipboard()
                    self.insertPlainText(clipboard.text())
                else:
                    super().keyPressEvent(event)
            
        self.display_area = MyTextEdit(self)
        self.display_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
      
        self.mainFont = QFont("Noto Color Emoji", 14)
        self.display_area.setFont(self.widgetFont)
        self.display_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
        text_layout.addWidget(self.display_area)
        # Control Panel
        control_layout = QVBoxLayout()
      
        # Buttons and Comboboxes
        self.discuss_button = QPushButton("discuss")
        self.discuss_button.setFont(self.widgetFont)
        self.discuss_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
        self.discuss_button.clicked.connect(self.discuss)
        control_layout.addWidget(self.discuss_button)
      
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)
        self.show()
        
    def display_msg(self, r):
        self.display_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        r = str(r)
        # presumably helps handle extended chars
        encoded = self.codec.fromUnicode(r)
        decoded = encoded.data().decode('utf-8')+'\n'
        self.display_area.insertPlainText(decoded)  # Insert the text at the cursor position
        self.display_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        self.display_area.repaint()

    def discuss(self):
        discuss_resources(self, self.query, self.sections, self.dscp, self.template)

def discuss_resources(display, query, sections, dscp='', template=None):
    """ resources is [[section_ids], [[title, section_synopsis], ...]]"""
    # excerpts is just [[title, section_synopsis], ...]
    global updated_json, config

    section_ids = sections[0]
    excerpts = sections[1]
    config={}
    rows = ["Outline", "Search", "Write", "ReWrite"]
    # make dummy config, write_aux needs it
    for row in rows:
        config[row] = {"model":'llm', "exec":'Yes'}

    
    # now enter conversation loop.
    while query is not None:
        length = 600 # default response - should pick up from ui max_tokens if run from ui, tbd
        #create detail instruction for response from task, sbar, query
        instruction_prompt = """Generate a concise instruction for a research assistant on what to extract from a set of research papers to answer the Query below, given the Topic and Background information provided.
Respond using this JSON format:
{"instruction": '<instruction>'}

Topic: {{$topic}}
Background: {{$dscp}}
Query: {{$query}}

Respond only with the instruction, in plain JSON as above, with no markdown or code formatting.
"""
        messages = [SystemMessage(cot.v_short_prompt()),
                    UserMessage(instruction_prompt),
                    AssistantMessage('')
                    ]
        instruction = cot.llm.ask({"topic":dscp, "query":query, "dscp": dscp},
                                  messages, stop_on_json=True, max_tokens=250, validator=JSONResponseValidator())
        outline = {"title": query, "rewrites": 2, "length": length, "dscp": str(instruction['instruction'])}
        display.display_msg(json.dumps(outline, indent=2))
        
        if len(excerpts) > 10: 
            # s2_search_resources(outline, resources)
            pass

        report, refs = write_report_aux(config, paper_outline=outline, section_outline=outline, heading_1_title=query, length=600, resources=resources)

        display.display_msg(report)
        display.display_msg(refs)
        query = cot.confirmation_popup('Query?', '')
        if query is None or not query:
            return report, refs 
    
if __name__ == '__main__':
    def parse_arguments():
        """Parses command-line arguments using the argparse library."""
        
        parser = argparse.ArgumentParser(description="Process a single command-line argument.")
        
        parser.add_argument("-discuss", type=str, help="discuss a provided set of resources")
        parser.add_argument("-report", type=str, help="discuss a provided set of resources")
        args = parser.parse_args()
        return args
    try:
        args = parse_arguments()
        if hasattr(args, 'discuss') and args.discuss is not None:
            print(args.discuss)
            with open(args.discuss, 'rb') as f:
                resources = pickle.load(f)
            print(f'\nResources:\n{json.dumps(resources, indent=2)}')
            app = QApplication(sys.argv)
            window = DisplayApp(input('?'), resources['sections'], resources['dscp'], resources['template'])
            #window.show()
            app.exec()
            sys.exit(0)
        if hasattr(args, 'report') and args.report is not None:
            write_report(app, args.report)
            sys.exit(0)
        else:
            print('paper_writer.py -report expects to be called from Owl with topic')
            pass
    except Exception as e:
        traceback.print_exc()
        print(str(e))
        sys.exit(-1)
    
