import os, sys, logging, glob, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import argparse
import requests
import urllib.request
import numpy as np
import pandas as pd
import pickle
import subprocess
import json
import re
import random
import openai
import traceback
from pathlib import Path
import arxiv
from arxiv import Client, Search, SortCriterion, SortOrder
import ast
import concurrent
from csv import writer
#from IPython.display import display, Markdown, Latex
from lxml import etree
from PyPDF2 import PdfReader
import pdfminer.high_level as miner
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import faiss
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from termcolor import colored
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.Prompt import Prompt
from promptrix.GPT3Tokenizer import GPT3Tokenizer
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
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication, QCheckBox
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem, QLineEdit
from OwlCoT import LLM, ListDialog, generate_faiss_id
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from transformers import AutoTokenizer, AutoModel
import webbrowser
import rewrite as rw
import grobid
# used for title matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests, json
from lxml import etree
url = "http://192.168.1.160:8070/api/processFulltextDocument"
pdf_filepath  = "./arxiv/papers/A_3-miRNA_signature_predicts_prognosis_of_pediatric_and_adolescent_cytogenetically_normal_acute_myeloid_leukemia"
#pdf_filepath = "./arxiv/papers/2311.17970.pdf"

headers = {"Response-Type": "application/xml"}#, "Content-Type": "multipart/form-data"}
# Define the namespace map to handle TEI namespaces
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

def process_xml_table(table):
    start_row_index = -1
    # Try to extract headers using <th> elements; if not present, use the first row's <td> elements
    th_elements = table.xpath('.//tei:tr[1]/tei:th', namespaces=ns)
    if th_elements and len(th_elements)>0 :
        headers = [th.text for th in th_elements]
        start_row_index = 1  # Start from the first row for data rows if headers were found in <th>
    else:
        # If no <th> elements, use the first row's <cell> for headers
        headers = [cell.text for cell in table.xpath('.//tei:row[1]/tei:cell', namespaces=ns)]
        start_row_index = 2  # Start from the second row for data rows, since headers are in the first
        
    #print(f'start_row {start_row_index} headers {headers}')
    # Initialize an empty list to hold each row's data as a dictionary
    table_data = []
        
    # Iterate over each row in the table, excluding the header row
    for row in table.xpath(f'.//tei:row[position()>={start_row_index}]', namespaces=ns):
        # Extract text from each cell (<td>) in the row
        row_data = [cell.text for cell in row.xpath('.//tei:cell', namespaces=ns)]
        if len(row_data) != len(headers): # don't try to process split rows
            continue
        # Create a dictionary mapping each header to its corresponding cell data
        row_dict = dict(zip(headers, row_data))
        table_data.append(row_dict)
            
    # Convert the table data to JSON
    if len(table_data) >0 : # don't store tables with no rows (presumably result of no rows matching headers len)
        #json_data = json.dumps(table_data, indent=4)
        return table_data
    else:
        return None

def parse_pdf(pdf_filepath):
    print(f'grobid parse_pdf {pdf_filepath}')
    max_section_len = 3072 # chars, about 1k tokens
    files= {'input': open(pdf_filepath, 'rb')}
    extract = {"title":'', "authors":'', "abstract":'', "sections":[], "tables": []}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        with open('test.tei.xml', 'w') as t:
            t.write(response.text)
    else:
        print(f'grobid error {response.status_code}, {response.text}')
        return None
    # Parse the XML
    xml_content = response.text
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.fromstring(xml_content.encode('utf-8'))
    # Extract title
    title = tree.xpath('.//tei:titleStmt/tei:title[@type="main"]', namespaces=ns)
    title_text = title[0].text if title else 'Title not found'
    extract["title"]=title_text

    # Extract authors
    authors = tree.xpath('.//tei:teiHeader//tei:fileDesc//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:persName', namespaces=ns)
    authors_list = [' '.join([name.text for name in author.xpath('.//tei:forename | .//tei:surname', namespaces=ns)]) for author in authors]
    extract["authors"]=', '.join(authors_list)

    # Extract abstract
    abstract = tree.xpath('.//tei:profileDesc/tei:abstract//text()', namespaces=ns)
    abstract_text = ''.join(abstract).strip()
    extract["abstract"]=abstract_text

    # Extract major section titles
    # Note: Adjust the XPath based on the actual TEI structure of your document
    section_titles = tree.xpath('./tei:text/tei:body/tei:div/tei:head', namespaces=ns)
    titles_list = [title.text for title in section_titles]
    body_divs = tree.xpath('./tei:text/tei:body/tei:div', namespaces=ns)
    pp_divs = tree.xpath('./tei:text/tei:body/tei:p', namespaces=ns)
    figures = tree.xpath('./tei:text/tei:body/tei:figure', namespaces=ns)
    pdf_tables = []
    for figure in figures:
        # Retrieve <table> elements within this <figure>
        # Note: Adjust the XPath expression if the structure is more complex or different
        tables = figure.xpath('./tei:table', namespaces=ns)
        if len(tables) > 0:
            pdf_tables.append(process_xml_table(tables[0]))
    extract['tables'] = pdf_tables
    sections = []
    for element in body_divs:
        #print(f"\nprocess div chars: {len(' '.join(element.xpath('.//text()')))}")
        head_texts = element.xpath('./tei:head//text()', namespaces=ns)
        all_text = element.xpath('.//text()')
        # don't need following anymore, concatenating text segments with '\n'
        #for head in head_texts:
        #    #print(f'  process head {head}')
        #    for t, text in enumerate(all_text):
        #        if head == text:
        #            #print(f'  found head  in all_text {t}')
        #            all_text[t] = head+'\n'

        # Combine text nodes into a single string
        combined_text = '\n'.join(all_text)
        if len(combined_text) < 24:
            continue
        if len(combined_text) > max_section_len: #break into chunks on pp
            pps = ''
            for text in all_text:
                if len(pps) + len(text) < max_section_len:
                    pps += '\n'+text
                elif len(pps) > int(.5* max_section_len):
                    sections.append(pps)
                    pps = text
                else:
                    sections.append(pps+'\n'+text)
                    pps = ''
        else:
            sections.append(combined_text)
                    
    extract["sections"] = sections
    print(f'title: {title_text}')
    #print(f"Abstract: {len(abstract_text)} chars, Section count: {len(body_divs)}, tables: {len(pdf_tables)}, max_section_len: {max_section_len}")
    return extract

def get_title(title):
    title = title.strip()
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=url,title,year,abstract,authors,citationStyles,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,s2FieldsOfStudy"
        headers = {'x-api-key':ssKey, }
        response = requests.get(url, headers = headers)
        if response.status_code != 200:
            print(f'SemanticsSearch fail code {response.status_code}')
            return None
        results = response.json()
        #print(f' s2 response keys {results.keys()}')
        total_papers = results["total"]
        if total_papers == 0:
            return None
        papers = results["data"]
        #print(f'get article search returned first {len(papers)} papers of {total_papers}')
        for paper in papers:
            paper_title = paper["title"].strip()
            print(f'considering {paper_title}')
            if paper_title.startswith(title):
                #data = get_semantic_scholar_meta(paper['paperId'])
                print(f'title meta-data {paper.keys()}')
                return paper
    except Exception as e:
        traceback.print_exc()

def reform_strings(strings, min_length=32, max_length=2048):
    """
    Combine sequences of shorter strings in the list, ensuring that no string in the resulting list 
    is longer than max_length characters, unless it existed in the original list.
    also discards short strings.
    """
    combined_strings = []
    current_string=''
    for string in strings:
        if len(string) < 24:
            continue
        if not current_string:
            current_string = string
        elif len(current_string) + len(string) > max_length:
            combined_strings.append(current_string)
            current_string = string
        else:
            current_string += ('\n' if len(current_string)>0 else '') + string
    if current_string:
        combined_strings.append(current_string)
    return combined_strings
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def index_paper(pdf_filepath):
    # main path for papers from index_url and / or index_file, where we know nothing other than pdf
    try:
        paper_dict = {}
        extract = parse_pdf(pdf_filepath)
        if extract is None:
            print('grobid extract is None')
            return False
        print(f'index_paper grobid keys {extract.keys()}')
        title = extract['title']
        paper_dict['title'] = extract['title']
        paper_dict['authors'] = extract['authors']
        paper_dict['summary'] = extract['abstract'] # will be overwritten by S2 abstract if we can get it
        meta_data = get_title(title)
        if meta_data is not None:
            paper_dict["summary"] = str(meta_data['abstract'])
            #paper_dict["year"] = meta_data['year'])
            paper_dict["article_url"] = str(meta_data['url'])
            paper_dict["summary"] = str(meta_data['abstract'])
            paper_dict["citationCount"]= int(meta_data['citationCount'])
            paper_dict["citationStyles"]= str(meta_data['citationStyles'])
            paper_dict["inflCitations"] = int(meta_data['influentialCitationCount'])
        abstract = extract['abstract']
        
    except Exception as e:
        print(f'\ngrobid fail {str(e)}')
        return False
    print(f"grobid extract keys {extract.keys()}")
    text_chunks = [abstract]+extract['sections']
    print(f"Summarizing each chunk of text {len(text_chunks)}")

    section_prompt = """Given this abstract of a paper:
    
<ABSTRACT>
{{$abstract}}
</ABSTRACT>

and this list of important entities (key phrases, acronyms, and named-entities) mentioned in this section:

<ENTITIES>
{{$entities}}
</ENTITIES>

Generate a synposis of this section of text from the paper. The section synopsis should include the central argument of the section as it relates to the abstract and previous sections, together with all key points supporting that central argument. A key point might be any observation, statement, fact, inference, question, hypothesis, conclusion, or decision relevant to the central argument of the text. The section synopsis should be 'entity-dense', that is, it should include all relevant ENTITIES to the central argument of the text and the paper abstract.

<TEXT>
{{$text}}
</TEXT>

End your synopsis with:
</SYNOPSIS>
"""

    section_synopses = []; section_ids = []
    #text_chunks = combine_strings(text_chunks) # combine shorter chunks
    rw_count = 0
    for idx, text_chunk in enumerate(text_chunks):
        if len(text_chunk) < 32:
            continue
        rw_count += 1
        sentences = [text_chunk]
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
        model.eval()
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        ### no more rewriting on ingest,
        ###   worried it distorts content more than content uniformity of style aids downstream extract
        
        id = index_section_synopsis(paper_dict, text_chunk)
        section_synopses.append(text_chunk)
        section_ids.append(id)
        
if __name__ == '__main__':
    import OwlCoT as CoT
    template = None
    cot = CoT.OwlInnerVoice(None)
    rw.cot = cot
    #extract = parse_pdf(pdf_filepath)
    app = QApplication(sys.argv)
    rewrite = index_paper(pdf_filepath)
    #print(f'{json.dumps(extract, indent=2)}\n{len(json.dumps(extract, indent=2))}')
