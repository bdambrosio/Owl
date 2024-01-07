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
from OwlCoT import LLM, ListDialog, generate_faiss_id,OPENAI_MODEL3,OPENAI_MODEL4
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from transformers import AutoTokenizer, AutoModel
import webbrowser
from Planner import Planner
import rewrite as rw

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
# set cot for rewrite so it can access llm
rw.cot = cot

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

def get_template(row, config):
    if row in config:
        if 'model' in config[row]:
            model = config[row]['model']
            if model == 'gpt3':
                return OPENAI_MODEL3
            if model == 'gpt4':
                return OPENAI_MODEL4
            if model == 'llm':
                return cot.template
        else:
            return cot.template
    print(f'get_template fail {row}\n{json.dumps(config, indent=2)}')
        
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

def entities(paper_title, paper_outline, paper_summaries, ids,template):
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
            excerpt_items = extract_entities(paper_title, paper_outline, excerpt[0]+'\n'+excerpt[1], template)
            entity_cache[int_id]=excerpt_items
            items.extend(entity_cache[int_id])
    print(f'entities total {total}, in cache: {cached}')
    with open(entity_cache_filepath, 'w') as pf:
        json.dump(entity_cache, pf)
    print(f"wrote {entity_cache_filepath}")
    return list(set(items))

def extract_entities(paper_title, paper_outline, summary, template):
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

Respond in a plain JSON format without any Markdown or code block formatting, and without any comments or explanatory text, using the following format:
{{"found": ["keywd1", "Named Entity1", "Acronym1", "keywd2", ...]}}
"""),
                  AssistantMessage("")
              ]
    
    response_json = cot.llm.ask('', kwd_messages, template=template, max_tokens=400, temp=0.1, stop_on_json=True, validator=JSONResponseValidator())
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

def make_search_queries(outline, section_outline, template):
    prompt = """
Following is an outline for a research paper, in JSON format: 

{{$outline}}

From this outline generate 3 SemanticScholar search queries for the section:
{{$target_section}}

Respond in a plain JSON format without any Markdown or code block formatting,  using the following format:
{"query1":'query 1 text',"query2": 'query 2 text', "query3": query 3 text'}

Respond ONLY with the JSON above, do not include any commentary or explanatory text.
"""
    messages = [SystemMessage(prompt),
                AssistantMessage('')
                ]
    queries = cot.llm.ask({"outline":outline, "target_section":section_outline}, messages, stop_on_json=True, template=template, max_tokens=150, validator=JSONResponseValidator())
    if type(queries) is dict:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{json.dumps(queries, indent=2)}')
    else:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{queries}')
    return queries
    
def s2_search (config, outline, section_outline):
    #
    ## call get_articles in semanticScholar2 to query semanticScholar for each section or subsection in article
    ## we can get 0 results if too specific, so probabilistically relax query as needed
    #
    print(f'paper_writer entered s2_search with {section_outline}')
    template = get_template('Search', config)
    print(f' paper_writer s2_search template: {template}')
    # llms sometimes add an empty 'sections' key on leaf sections.
    if 'sections' in section_outline and len(section_outline['sections']) > 0: 
        for subsection in section_outline['sections']:
            s2_search(config, outline, subsection)
    else:
        queries = make_search_queries(outline, section_outline, template)
        bads = ['(',')',"'",' AND', ' OR'] #tulu sometimes tries to make fancy queries, S2 doesn't like them
        if type(queries) is dict:
            for i in range(3):
                if 'query'+str(i) in queries:
                    query = queries['query'+str(i)]
                    for bad in bads:
                        query = query.replace(bad, '')
                        result_list, total_papers, next_offset = s2.get_articles(query, confirm=True)
                        print(f's2_search found {len(result_list)} new papers')

                
def write_report(config, length=None): 
    #rows = ["Query", "SBAR", "Outline", "WebSearch", "Write", "ReWrite"]
    query_config = config['Query']
    query = ""
    plan = plan_search()
    if query_config['exec'] == 'Yes':
        query = cot.confirmation_popup('Question to report on?', '')
    else:
        plan['task']=query
            
    sbar_config = config['SBAR']
    if sbar_config['exec'] == 'Yes':
        # pass in query to sbar!
        plan = pl.analyze(plan)
        
    outline_config = config['Outline']
    if outline_config['exec'] == 'Yes':
        # make outline
        pl.outline(outline_config, plan)
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
        if length is None:
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


def write_report_aux(config, paper_outline=None, section_outline=None, length=400, inst='', topic='', paper_title='', abstract='', depth=0, parent_section_title='', parent_section_partial='', heading_1=None, heading_1_draft = '', num_rewrites=1):
    template = get_template('Write',config)
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
    if 'sections' in section_outline and len(section_outline['sections']) > 0:
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
        ids, excerpts = s2.search(query) # this assumes web searching has been done note we aren't using dscp for search
        paper_summaries = '\n'.join(['Title: '+s[0]+'\n'+s[1] for s in excerpts])
        subsection_token_length = max(320,length) # no less than a paragraph
        
        #
        ### Write initial content
        #
        
        print(f"\nWriting:{section_outline['title']} length {length}\n  within {parent_section_title}\n covering {subsection_topic}")
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


The {heading_1} section content up to this point is:

<{heading_1}_SECTION_CONTENT>
{heading_1_draft}
</{heading_1}_SECTION_CONTENT>

Again, your current task is to write the next part, titled: '{section_outline["title"]}'

1. First reason step by step to determine the role of the part you are generating within the overall paper, 
2. Then generate the appropriate text, subject to the following guidelines:

 - Output ONLY the text, do NOT output your reasoning.
 - Write a dense, detailed text using known fact and the above research excepts, of about {subsection_token_length} words in length.
 - This section should cover the specific topic: {parent_section_title} - {subsection_topic}
 - You may refer to, but not repeat, prior section content in any text you produce.
 - Present an integrated view of the assigned topic, noting controversy where present, and capturing the overall state of knowledge along with essential statements, methods, observations, inferences, hypotheses, and conclusions. This must be done in light of the place of this section or subsection within the overall paper.
 - Ensure the section provides depth while removing redundant or superflous detail, ensuring that every critical aspect of the source argument, observations, methods, findings, or conclusions is included.
End the section as follows:

</DRAFT>
"""),
              AssistantMessage("<DRAFT>\n")
              ]
        response = cot.llm.ask('', messages, template=template, max_tokens=subsection_token_length, temp=0.1, eos='</DRAFT>')
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
        keywds = entities(paper_title, paper_outline, excerpts, ids, template)
        missing_entities = literal_missing_entities(keywds, draft)
        print(f'\n missing entities in initial draft {len(missing_entities)}\n')
        for i in range(num_rewrites):
            if i < num_rewrites-1:
                template = template
            else:
                template = get_template('ReWrite', config)
            if i < num_rewrites-1:
                #add new entities
                draft = rw.add_pp_rewrite(paper_title, section_outline['title'], draft, paper_summaries, keywds, subsection_topic, int((1.3**(i+1))*subsection_token_length), parent_section_title, heading_1, heading_1_draft, template)
            else:
                # refine in final rewrite
                draft = rw.rewrite(paper_title, section_outline['title'], draft, paper_summaries, keywds, subsection_topic, 2*subsection_token_length, parent_section_title, heading_1, heading_1_draft, template)
            missing_entities = literal_missing_entities(keywds, draft)
            print(f'\n missing entities after rewrite {len(missing_entities)} \n')

        # make sure we write out top level sections even if they have no subsections!
        if depth==0:
            with open(f'section{n}.txt', 'w') as pf:
                pf.write(section)
            n += 1
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
    ex = PWUI(rows, config) # get config for this report task
    app.exec_()
    try:
        cot.display_response('calling paper_writer')
        paper = write_report(config)
    except Exception as e:
        traceback.print_exc()
        print(str(e))
        sys.exit(-1)
    #print(f'\n\n\n\n{paper}')
    #with open('paper.txt', 'w') as pf:
    #    pf.write(paper)
    #print(json.dumps(outline, indent=2))
    
