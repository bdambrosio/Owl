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
import OwlCoT as oiv
from pyqt_utils import ListDialog, generate_faiss_id
import semanticScholar3 as s2
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
print (cot.template)
# set cot for rewrite so it can access llm
rw.cot = cot
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
                ##print(json.dumps(plan, indent=4))
                plans[plan['name']] = plan

    if plan is None:
        # new plan, didn't select any existing
        plan = pl.initialize()
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
{"query0":'query 0 text',"query1": 'query 1 text', "query2": query 2 text'}

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

def generate_s2_search_queries(query):
    information_requirements = cot.script.sufficient_response(query)
    prompt = """
Generate a set of short Semantic Scholar (s2) search queries, in JSON format, given the following set of information requirements withing the overall topic of:

{{$query}}

<InformationNeeds>
{{$needs}}
</InformationNeeds>

A query should contain no more than 10 tokens. 
Respond in a plain JSON format without any Markdown or code block formatting,  using the following format:

{"information-need1":'s2 query text 1',"information-need2": '<s2 query text 2>', ...}

Respond ONLY with the JSON above, do not include any commentary or explanatory text.
"""
    messages = [SystemMessage(prompt),
                AssistantMessage('')
                ]
    queries = cot.llm.ask({"query":query, "needs":information_requirements}, messages, stop_on_json=True, max_tokens=300, validator=JSONResponseValidator())
    if type(queries) is dict:
        print(f'\nquery: {query}\ns2 queries:\n{json.dumps(queries, indent=2)}')
    else:
        print(f'\nquery: {query}\ns2 queries:\n{queries}')
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
            while next_offset < total_papers and cot.confirmation_popup('Continue?', query):
                result_list, total_papers, next_offset = s2.get_articles(query, next_offset, confirm=True)
                
# global passed into jsonEditor widget
updated_json = None

def handle_json_editor_closed(result):
    global updated_json
    #print(f'got result {result}')
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
    if query_config['exec'] == 'Yes' or 'task' not in plan.keys():
        query = cot.confirmation_popup('Question to report on?', '')
    if 'task' not in plan.keys():
        plan['task']=query
    save_plans()
            
    sbar_config = config['SBAR']
    if sbar_config['exec'] == 'Yes':
        # pass in query to sbar!
        #print(f'sbar input plan\n{plan}')
        if 'sbar' in plan and type(plan['sbar']) is dict:
            if cot.confirmation_popup('Edit existing sbar?', json.dumps(plan['sbar'], indent=2)):
                # we already have an sbar, edit it
                app = QApplication(sys.argv)
                editor = ew.JsonEditor(plan['sbar'])
                editor.closed.connect(handle_json_editor_closed)
                editor.show()
                app.exec()
                print(f'SBAR: {updated_json}')
                if updated_json is not None: # jsonEditor returns None if user presses cancel
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
            outline = plan['outline']
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
    #
    ## need to mod this to handle 'resources' longer than available context
    #
    template = get_template('Write',config)
    if depth == 0: 
        n = 0; refs=[] #set section number initially to 0
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
    if depth == 0: 
        if resources is not None:
            ids = resources[0]
            excerpts = resources[1]
            paper_summaries = '\n'.join(['Title: '+s[0]+'\n'+s[1] for s in excerpts])
            print(f'write_report_aux depth 0 len {len(paper_summaries)}, limit {cot.llm.context_size*2} chars')
            if len(paper_summaries) > cot.llm.context_size*2: # context is in tokens, len is chars
                # excerpts are too long for context, do query-specific extract first
                rewrites = []
                for id, excerpt in zip(ids, excerpts):
                    title = excerpt[0]
                    text = excerpt[1]
                    if len(text) < (cot.llm.context_size*1.5)/len(paper_summaries):
                        # short section, don't bother with extract
                        rewrites.append([title, text])
                        continue
                    row = s2.get_section(id)
                    if row is not None:
                        ners = row['ners']
                    else:
                        ners = ' '.join(rw.extract_ners(subsection_dscp+'. '+text))
                    tokens = int(((1.0*len(text))/len(paper_summaries))*(cot.llm.context_size*2))
                    rewrite = rw.extract(title, text, draft=None,
                                         instruction=subsection_dscp,
                                         ners=ners, tokens=tokens, template=template)
                    print(f'\nWRAUX in {len(text)} out {len(rewrite)}\n{rewrite}\n')
                    if len(rewrite) < len(text):
                        rewrites.append([title, rewrite])
                    else: # sometimes extract takes more words than the original, ignore it
                        rewrites.append([title, text])
                resources = [ids, rewrites]
            print('paper_writer rewritten resources:\n{json.dumps(resources, indent=2)}')


    if 'sections' in section_outline and len(section_outline['sections']) > 0:
        #
        ### write section intro first draft
        #
        subsection_depth = 1+depth
        num_sections = len(section_outline['sections'])
        subsection_token_length = int(length/len(section_outline['sections']))
        section = ''
        n=0
        refs = []
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
        section = '' if 'title' not in section_outline else section_outline['title']
        print(f'heading_1 {heading_1_title}\npst {parent_section_title}\nsubsection topic {subsection_topic}')
        query = heading_1_title+', '+parent_section_title+' '+subsection_topic
        # below assumes web searching has been done
        if resources is None:
            # do local search, excerpts to use not provided
            print(f'** write_report_aux Doing local search ! **')
            ids, excerpts = s2.search(query, subsection_dscp) 
        else:
            # resources ['sections'] = [[id1, id2, ...], [[title1, text1],[title2, text2],...]]
            ids = resources[0]
            excerpts = resources[1]
        paper_summaries = '\n'.join(['Title: '+s[0]+'\n'+s[1] for s in excerpts])
        print (f'wra resource paper summaries len after rewrite {len(paper_summaries)}')
        subsection_token_length = max(500,length) # no less than a paragraph
        subsection_refs =  []
        for title in [s[0]for s in excerpts]:
            if title not in subsection_refs:
                subsection_refs.append(title)

        print(f"\nWriting: {section_outline['title']} length {length}")
        draft = rw.write(paper_title, paper_outline, section_outline['title'], paper_summaries, subsection_topic, int(subsection_token_length), parent_section_title, heading_1_title, heading_1_draft, template)
        #print(f'\nFirst Draft:\n{draft}\n')
        if num_rewrites < 1:
            return draft, subsection_refs

        #
        ### Now do rewrites
        #
        ### first collect entities
        #
        keywds = rw.paper_ners(paper_title, paper_outline, excerpts, ids, template)
        missing_entities = rw.literal_missing_ners(keywds, draft)
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
            missing_entities = rw.literal_missing_ners(keywds, draft)
            print(f'\n missing entities after rewrite {len(missing_entities)} \n')

        
        section = draft
        # make sure we write out top level sections even if they have no subsections!
        if depth==0: # single top-level section with no subsections
            with open(f'section{n}.txt', 'w') as pf:
                pf.write(draft)
            n += 1

    return section, subsection_refs

class DisplayApp(QtWidgets.QWidget):
    def __init__(self, query, paper_id, sections=[], dscp='', template=''):
        super().__init__()
        self.query = query
        self.paper_id = paper_id
        self.paper_row = None
        search_result = s2.paper_library_df[s2.paper_library_df['faiss_id']==int(paper_id)]
        if len(search_result) > 0:
            self.paper_row = search_result.iloc[0]
            print(f'found paper {self.paper_row.title}')
        else:
            print(f"couldn't find paper {self.paper_id}")
        print(f'sections {type(sections)}, {len(sections)}\n{sections[0]}')
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
            
        self.mainFont = QFont("Noto Color Emoji", 14)

        self.display_area = MyTextEdit(self)
        self.display_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.display_area.setFont(self.widgetFont)
        self.display_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
        text_layout.addWidget(self.display_area)

        self.input_area = MyTextEdit(self)
        self.input_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.input_area.setFont(self.widgetFont)
        self.input_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
        text_layout.addWidget(self.input_area)
      
        # Buttons and Comboboxes
        self.discuss_button = QPushButton("discuss")
        self.discuss_button.setFont(self.widgetFont)
        self.discuss_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
        self.discuss_button.clicked.connect(self.discuss)
        text_layout.addWidget(self.discuss_button)
      
        self.describe_button = QPushButton("describe")
        self.describe_button.setFont(self.widgetFont)
        self.describe_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
        self.describe_button.clicked.connect(self.describe)
        text_layout.addWidget(self.describe_button)
      
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
        input_text = self.input_area.toPlainText()
        self.input_area.clear()
        print(f'calling discuss query{self.query} sections {len(self.sections)}, dscp {self.dscp}')
        discuss_resources(self, input_text, self.sections, self.query+', '+self.dscp, self.template)

    def describe(self):
        self.display_msg(self.paper_row.extract)
        print(f'calling describe sections: {len(self.sections[0])}')
        print(f'section ids: {self.sections[0]}')
        entities = set()
        for id, title_text in zip(self.sections[0], self.sections[1]):
            int_id = str(int(id))
            print(id, title_text[0])
            section_entities = rw.section_ners(id, title_text[1], self.template)
            entities.update(section_entities)
            print(section_entities)
        print(entities)
        print(f'paper_id {self.paper_id}')
        search_result = s2.paper_library_df[s2.paper_library_df['faiss_id']==self.paper_id]
        if len(search_result) > 0:
            self.paper_row = search_result.iloc[0]
            print(f'found paper {self.paper_row.title}')
        classification_prompt = [SystemMessage("Construct a minimal set of STEM NERs that covers the NERs provided below. The response should contain 5 to 8 NERs that collectively cover 90% of the provided NERs, with no or small overlap among the response NERs and minimal set-difference between the ontological space covered by the response NERs and the ontological space covered by the union of the user-provided NERs. Respond ONLY with the response NERs, followed by </END>."),
                                 UserMessage('User NERs:\n{{$ners}}'),
                                 AssistantMessage('Response NERs:\n')
                                 ]
        response = cot.llm.ask({'ners':entities}, classification_prompt)
        print(response)


def discuss_resources(display, query, paper_id, sections, dscp='', template=None):
    """ resources is [[section_ids], [[title, section_synopsis], ...]]"""
    # excerpts is just [[title, section_synopsis], ...]
    global updated_json, config

    config={}
    rows = ["Outline", "Search", "Write", "ReWrite"]
    # make dummy config, write_aux needs it
    for row in rows:
        config[row] = {"model":'llm', "exec":'Yes'}

    
    # now enter conversation loop.
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
    outline = {"title": query, "rewrites": 1, "length": length, "dscp": str(instruction['instruction']),
               "task": str(instruction['instruction'])}
    display.display_msg(json.dumps(outline, indent=2))
    
    report, refs = write_report_aux(config, paper_outline=outline, section_outline=outline, heading_1_title=query, length=600, resources=sections)
    
    display.display_msg(report)
    display.display_msg(refs)
    return report, refs 
    
from Planner import Planner

if __name__ == '__main__':
    generate_s2_search_queries('what is miRNA?')
    sys.exit(0)
    def parse_arguments():
        """Parses command-line arguments using the argparse library."""
        
        parser = argparse.ArgumentParser(description="Process a single command-line argument.")
        
        parser.add_argument("-discuss", type=str, help="discuss a provided set of resources")
        parser.add_argument("-report", type=str, help="discuss a provided set of resources")
        parser.add_argument("-template", type=str, help="llm prompt template")
        args = parser.parse_args()
        return args
    try:
        args = parse_arguments()
        if hasattr(args, 'discuss') and args.discuss is not None:
            print(args.discuss)
            with open(args.discuss, 'rb') as f:
                resources = pickle.load(f)
            #print(f'\nResources:\n{json.dumps(resources, indent=2)}')
            app = QApplication(sys.argv)
            cot = oiv.OwlInnerVoice(None, template=resources['template'])
            # set cot for rewrite so it can access llm
            rw.cot = cot
            s2.cot = cot
            #print(f'resources {json.dumps(resources, indent=2)}')
            print(f'paper_id {resources["papers"][0]}')
            #print(f'resources [0][1:] {resources["papers"][1]}')
            window = DisplayApp(query='Question?',
                                paper_id=resources['papers'][0],
                                sections=resources['papers'][1],
                                dscp=resources['dscp'],
                                template=resources['template'])
            #window.show()
            app.exec()
            sys.exit(0)

        if hasattr(args, 'report') and args.report is not None:
            if hasattr(args, 'template'):
                cot = oiv.OwlInnerVoice(None, args.template)
            else:
                cot = oiv.OwlInnerVoice(None)
            # set cot for rewrite so it can access llm
            rw.cot = cot
            s2.cot = cot

            pl = Planner(None, cot, args.template)
            app = QApplication(sys.argv)
            write_report(app, args.report)
            sys.exit(0)
        else:
            print('paper_writer.py -report expects to be called from Owl with topic')
            pass
    except Exception as e:
        traceback.print_exc()
        print(str(e))
        sys.exit(-1)
    
