import os, sys, logging, glob, time
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
import torch
import torch.nn.functional as F
from wordfreq import tokenize as wf_tokenize
from transformers import AutoTokenizer, AutoModel
import webbrowser
import rewrite as rw
import grobid
# used for title matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = GPT3Tokenizer()
ui = None
cot = None

# startup AI resources

#from adapters import AutoAdapterModel
#embedding_model = AutoAdapterModel.from_pretrained("/home/bruce/Downloads/models/Specter-2-base")
#embedding_adapter_name = embedding_model.load_adapter("/home/bruce/Downloads/models/Specter-2", set_active=True)
embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
embedding_model.eval()


GPT3 = "gpt-3.5-turbo"
GPT4 = "gpt-4"
ssKey = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
#logging.basicConfig(level=logging.DEBUG)
directory = './arxiv/'
# Set a directory to store downloaded papers
papers_dir = os.path.join(os.curdir, "arxiv", "papers")
paper_library_filepath = "./arxiv/paper_library.parquet"

section_index_filepath = "./arxiv/section_index_w_idmap.faiss"
section_indexIDMap = None

paper_index_filepath = "./arxiv/paper_index_w_idmap.faiss"
paper_indexIDMap = None

# section library links section embedding faiss ids to section synopsis text and paper library (this latter through paper url)
section_library_filepath = "./arxiv/section_library.parquet"
section_library = None


# Check for and create as needed main arxiv directory and papers directory
if not os.path.exists(directory):
    # If the directory doesn't exist, create it and any necessary intermediate directories
    os.makedirs(directory)
    os.makedirs(papers_dir)
    print(f"Directory '{directory}' created successfully.")
else:
    print(f"Directory '{directory}' exists.")
    if not os.path.exists(papers_dir):
        print(f"creating Directory '{papers_dir}'.")
        os.makedirs(papers_dir)
    else:
        print(f"Directory '{papers_dir}' exists.")
        

def get_semantic_scholar_meta(s2_id):
    #url = f"https://api.semanticscholar.org/v1/paper/doi:{doi}"
    url = f"https://api.semanticscholar.org/v1/paper/{s2_id}?fields=url,title,year,abstract,authors,citationStyles,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,publisher,citationStyles"
    headers = {'x-api-key':ssKey}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to retrieve data for arXiv ID:", arxiv_id)
        return None

def paper_library_df_fixup():
    #paper_library_df['citation'] = ''
    #paper_library_df.to_parquet(paper_library_filepath)
    return

paper_library_columns = ["faiss_id", "title", "authors", "citationStyles", "publisher", "summary", "inflCitations", "citationCount", "article_url", "pdf_url","pdf_filepath", "section_ids"]
paper_library_init_values = {"faiss_id":0, "title":'', "authors":'', "citationStyles":'', "publisher":'', "summary":'', "inflCitations":0, "citationCount":0, "article_url":'', "pdf_url":'',"pdf_filepath":'', "synopsis":'', "section_ids":[]}

def validate_paper(paper_dict):
    #print(json.dumps(paper_dict, indent=2))
    if type(paper_dict) is not dict:
        print(f'invalid paper_dict: not a dict!')
        return False
    for field in paper_library_columns:
        if field not in paper_dict:
            print(f'field missing in paper_dict: {field}')
            return False
    if paper_dict['faiss_id'] == 0:
        print(f'invalid faiss_id in paper_dict: {paper_dict["faiss_id"]}')
        return False
    if len(paper_dict['title']) < 2:
        print(f'missing title in paper_dict: {paper_dict["title"]}')
        return False
    if len(paper_dict['summary']) < 16:
        print(f'missing abstract in paper_dict: {paper_dict["summary"]}')
        return False
    if len(paper_dict['pdf_url']) < 2 and len(paper_dict['pdf_filepath']) < 2:
        print(f'missing pdf path in paper_dict: {paper_dict["pdf_url"]}, {paper_dict["pdf_filepath"]}')
        return False
    if len(paper_dict['article_url']) < 2:
        print(f'missing article url in paper_dict: {paper_dict["article_url"]}')
        return False
    return True
        
# s2FieldsOfStudy
s2FieldsOfStudy=["Computer Science","Medicine","Chemistry","Biology","Materials Science","Physics",
                 "Geology","Psychology","Art","History","Geography","Sociology","Business","Political Science",
                 "Economics","Philosophy","Mathematics","Engineering","Environmental Science","Agricultural and Food Sciences",
                 "Education","Law","Linguistics"]
if not os.path.exists(paper_library_filepath):
    # Generate a blank dataframe where we can store downloaded files
    paper_library_df = pd.DataFrame(columns=paper_library_columns)
    paper_library_df.to_parquet(paper_library_filepath)
    print('paper_library_df.to_parquet initialization complete')
else:
    paper_library_df = pd.read_parquet(paper_library_filepath)
    paper_library_df_fixup()
    print('loaded paper_library_df')
    
if not os.path.exists(paper_index_filepath):
    paper_indexIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(768))
    faiss.write_index(paper_indexIDMap, paper_index_filepath)
    print(f"created '{paper_index_filepath}'")
else:
    paper_indexIDMap = faiss.read_index(paper_index_filepath)
    print(f"loaded '{paper_index_filepath}'")
    
if not os.path.exists(section_index_filepath):
    section_indexIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(768))
    faiss.write_index(section_indexIDMap, section_index_filepath)
    print(f"created '{section_index_filepath}'")
else:
    section_indexIDMap = faiss.read_index(section_index_filepath)
    print(f"loaded '{section_index_filepath}'")
    
if not os.path.exists(section_library_filepath):
    # Generate a blank dataframe where we can store downloaded files
    section_library_df =\
        pd.DataFrame(columns=["faiss_id", "paper_id", "ners", "ner_faiss_id", "synopsis"])
    section_library_df.to_parquet(section_library_filepath)
    print(f"created '{section_library_filepath}'")
else:
    section_library_df = pd.read_parquet(section_library_filepath)
    print(f"loaded '{section_library_filepath}'\n  keys: {section_library_df.keys()}")

def save_synopsis_data():
    paper_library_df.to_parquet(paper_library_filepath)
    faiss.write_index(paper_indexIDMap, paper_index_filepath)
    faiss.write_index(section_indexIDMap, section_index_filepath)
    section_library_df.to_parquet(section_library_filepath)

def get_section(id):
    section_library_rows = section_library_df[section_library_df['faiss_id'] == id]
    if len(section_library_rows) < 1:
        section_library_rows = section_library_df[section_library_df['ner_faiss_id'] == id]
    if len(section_library_rows) > 0:
        section_row = section_library_rows.iloc[0]
    return section_row

def search_sections(query, top_k=20):
    #print(section_library_df)
    # get embed
    query_embed = embedding_request(query, 'search_query: ')
    # faiss_search
    embeds_np = np.array([query_embed], dtype=np.float32)
    scores, ids = section_indexIDMap.search(embeds_np, top_k)
    #print(f'ss ids {ids}, scores {scores}')
    # lookup text in section_library
    if len(ids) < 1:
        print(f'No items found')
        return [],[]
    item_ids=[]; synopses = []
    for id in ids[0]:
        section_library_rows = section_library_df[section_library_df['faiss_id'] == id]
        if len(section_library_rows) < 1:
            section_library_rows = section_library_df[section_library_df['ner_faiss_id'] == id]
        if len(section_library_rows) > 0:
            section_row = section_library_rows.iloc[0]
            text = section_row['synopsis']
            filtered_df = paper_library_df[paper_library_df['faiss_id'] == section_row['paper_id']]
            if filtered_df.empty:
                article_title = 'unknown'
            else:
                paper_row = filtered_df.iloc[0]
                article_title = paper_row['title']
            item_ids.append(id)
            synopses.append([article_title, text])
            #print(f'{article_title} {len(text)}')
    return item_ids, synopses

def convert_title_to_unix_filename(title):
    filename = title.replace(' ', '_')
    # Remove or replace special characters
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
    filename = filename
    return filename

def download_pdf(url, title=None):
    #print(f'download_pdf {url}')
    if not title:
        title = url
    if url.startswith('file:///'):
        return url[len('file://'):]
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request) as response:
            pdf_data = response.read()
            filename = convert_title_to_unix_filename(title)
            arxiv_filepath = os.path.join(papers_dir, filename)
            with open(arxiv_filepath, 'wb') as f:  # Open in write binary mode
                f.write(pdf_data)
        print(f' download got paper, wrote to {arxiv_filepath}')
        return arxiv_filepath
    except Exception as e:
        print(f'\n download 1 fail {str(e)}')
    try:
        print(f'\n download_pdf trying 5005 retrieve')
        filepath=download_pdf_5005(url, title)
        print(f' download got paper, wrote to {filepath}')
        if filepath is not None:
            return filepath
    except Exception as e:
        print(f'\n download 5005 fail {str(e)}')
    try:
        print(f'\n download_pdf trying wbb')
        filepath = download_pdf_wbb(url, title)
        print(f' download got paper, wrote to {filepath}')
        return filepath
    except Exception as e:
        print(f'\n download wbb fail {str(e)}')
    return None

GOOGLE_CACHE_DIR = '/home/bruce/.cache/google-chrome/Default/Cache/Cache_Data/*'
def latest_google_pdf():
    try:
        files = [f for f in glob.glob(GOOGLE_CACHE_DIR)  if os.path.isfile(f)] # Get all file paths
        path = max(files, key=os.path.getmtime)  # Find the file with the latest modification time
        size = os.path.getsize(path)
        age = int(time.time()-os.path.getmtime(path))
        #print(f"secs old: {age}, size: {size}, name: {path}")
    except Exception:
        return '', 99999, 0
    return path, age, size

def wait_for_chrome(url, temp_filepath):
    found = False; time_left = 20
    while time_left >0:
        path, age, size = latest_google_pdf()
        #print(f'age {age}, size {size}, path {path}')
        if age < 2 and size > 1000:
            # now wait for completion - .5 sec with no change in size,we're done!
            prev_size = 0
            #print(f'\nwaiting for size to stablize\n')
            while prev_size < size:
                time.sleep(.5)
                prev_size = size
                path, age, size = latest_google_pdf()
            os.rename(path, temp_filepath)
            #print(f'\nGot it!\n')
            return temp_filepath
        time.sleep(.5)
        time_left = time_left-0.5
    return False
        
def download_pdf_wbb(url, title):
    global papers_dir
    try:
        for filename in os.listdir(GOOGLE_CACHE_DIR):
            file_path = os.path.join(GOOGLE_CACHE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except:
        pass

    # Send GET request to fetch PDF data
    # print(f' fetching url {url}')
    webbrowser.open(url)
    temp_filepath = os.path.join(papers_dir,'temp.pdf')
    new_filepath = wait_for_chrome(url, temp_filepath)
    #cot.confirmation_popup(title, 'Saved?' )
    if temp_filepath:
        filename = convert_title_to_unix_filename(title)
        arxiv_filepath = os.path.join(papers_dir, filename)
        os.rename(temp_filepath, arxiv_filepath)
        return arxiv_filepath
    else:
        print(f'download_pdf_wbb download attempt failed ')
    return None
        
def download_pdf_5005(url, title):
    global papers_dir
    # Send GET request to fetch PDF data
    #print(f' fetching url {url}')
    
    pdf_data = None
    try:
        response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}&doc_type=pdf', timeout=10)
    except Exception as e:
        print(f'download_pdf_5005 fail {str(e)}')
        return None
    if response.status_code == 200:
        info = response.json()
        #print(f'\nretrieve response {info};')
        if type(info) is dict and 'filepath' in info:
            idx = info['filepath'].rfind('/')
            arxiv_filepath = os.path.join(papers_dir, info['filepath'][idx+1:])
            os.rename(info['filepath'], arxiv_filepath)
            return arxiv_filepath
    else:
        print(f'paper download attempt failed {response.status_code}')
        return download_pdf_wbb(url, title)
    return None
        

def combine_strings(strings, min_length=32, max_length=2048):
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
    
#@retry(wait=wait_random_exponential(min=2, max=40), eos=stop_after_attempt(3))
def embedding_request(text, request_type='search_document: '):
    global embedding_tokenizer, embedding_model
    if type(text) != str:
        print(f'\nError - type of embedding text is not str {text}')
    text_batch = [request_type+' '+text]
    # preprocess the input
    encoded_input = embedding_tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = embedding_model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    #print(f'embedding_request response shape{embeddings.shape}')

    embedding = embeddings[0]
    # print(f'embedding_request response[0] shape{embedding.shape}')
    return embedding.detach().numpy()

# Function to recursively extract string values from nested JSON
# for getting text from sbar. Actually sbar is a flat structure, so we don't really need this much 'stuff'
def extract_string_values(data):
    string_values = []
    if isinstance(data, dict):
        for value in data.values():
            string_values.extend(extract_string_values(value))
    elif isinstance(data, list):
        for item in data:
            string_values.extend(extract_string_values(item))
    elif isinstance(data, str):
        string_values.append(data)
    return string_values

import re

# Function to extract words from JSON
def extract_words_from_json(json_data):
    string_values = extract_string_values(json_data)
    text = ' '.join(string_values)
    words = re.findall(r'\b\w+\b', text)
    return words


def index_dict(paper_dict):
    # called from index_service for search enqueued dict
    result = index_paper(paper_dict)
    if not result:
        return False
    save_synopsis_data()

def index_file(filepath):
    result_dict = {key: paper_library_init_values[key] for key in paper_library_columns}
    id = generate_faiss_id(lambda value: value in paper_library_df.faiss_id)
    result_dict["faiss_id"] = id
    result_dict["title"] = filepath[filepath.rfind('/')+1:] # will be replaced by grobid result
    result_dict["pdf_filepath"]= filepath
    result_dict['section_ids'] = [] # to be filled in after we get paper id
    # section and index paper
    result = index_paper(result_dict)
    if not result:
        return False
    #print(f"indexing new article  pdf file: {result_dict['pdf_filepath']}")
    save_synopsis_data()


def index_url(page_url):
    print(f'\nIndex URL {page_url}\n')
    result_dict = {key: paper_library_init_values[key] for key in paper_library_columns}
    id = generate_faiss_id(lambda value: value in paper_library_df.faiss_id)
    result_dict["faiss_id"] = id
    #result_dict["authors"] = authors
    result_dict["article_url"] = page_url
    result_dict["pdf_url"] = page_url
    if not page_url.startswith('file:///'):
        result_dict['pdf_url'] = page_url
        pdf_filepath= download_pdf(page_url)
    else:
        pdf_filepath= page_url[len('file://'):]
    result_dict["pdf_filepath"]= pdf_filepath
    
    # print(f" indexing article: {title}\n pdf file: {type(result_dict['pdf_filepath'])}")
    result_dict['synopsis'] = ""
    result_dict['section_ids'] = [] # to be filled in after we get paper id
    indexed = index_paper(result_dict)
    if not indexed:
        return False
    save_synopsis_data()



def get_arxiv_preprint_url(query, top_k=10):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It is used by s2 search when it can't download a title from S2 provided url.
    It also downloads the files and stores them in paper_library.csv to be retrieved by the read_article_and_summarize.
    """

    title_embed = embedding_request(query, 'search_query: ')
    result_list = []
    # Set up your search query
    # print(f'arxiv search for {query}')
    search = arxiv.Search(
        query=f'ti:{query}', 
        #query=query,
        max_results=top_k, 
        sort_by=SortCriterion.Relevance,
        sort_order = SortOrder.Descending
    )
    # Use the client to get the results
    results =arxiv.Client().results(search)
    # get closest embedding to title
    best_score = 0
    best_ppr = None
    for result in results:
        if len(paper_library_df) > 0:
            dup = (paper_library_df['title']== result.title).any()
            if dup:
                continue
        candidate_embed = embedding_request(result.title, 'search_document: ')
        cosine_similarity = np.dot(title_embed, candidate_embed) / (np.linalg.norm(title_embed) * np.linalg.norm(candidate_embed))
        #print(f' score {cosine_similarity}, title {result.title}')
        if cosine_similarity > best_score:
            best_ppr = result
            best_score = cosine_similarity
    if best_ppr is None:
        return None
    else:
        #print(f'considering {best_ppr.title}')
        if cot.confirmation_popup("found this, index it?", best_ppr.title):
            return [x.href for x in best_ppr.links][1]
        return None

def get_title(title):
    title = title.strip()
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=url,title,year,abstract,authors,citationStyles,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,s2FieldsOfStudy"
        headers = {'x-api-key':ssKey, }
        response = requests.get(url, headers = headers)
        if response.status_code != 200:
            print(f'SemanticSearch fail code {response.status_code}')
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


get_title("Can Large Language Models Understand Context")

def get_articles(query, next_offset=0, library_file=paper_library_filepath, top_k=10, confirm=True):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in paper_library.csv to be retrieved by the read_article_and_summarize.
    """

    # Initialize a client
    result_list = []
    #library_df = pd.read_parquet(library_file).reset_index()
    if confirm:
        query = cot.confirmation_popup("Search SemanticScholar using this query?", query )
    if not query:
        #print(f'confirmation: No!')
        return [],0,0
    else:
        # print(f'get_articles query: {query}')
        pass
    try:
        # Set up your search query
        #query='Direct Preference Optimization for large language model fine tuning'
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&offset={next_offset}&fields=url,title,year,abstract,authors,citationStyles,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,s2FieldsOfStudy,tldr"
        headers = {'x-api-key':ssKey, }
        
        response = requests.get(url, headers = headers)
        if response.status_code != 200:
            print(f'SemanticsSearch fail code {response.status_code}')
            return [],0,0
        results = response.json()
        #print(f' s2 response keys {results.keys()}')
        total_papers = results["total"]
        current_offset = results["offset"]
        if total_papers == 0:
            return [],0,0
        if current_offset == total_papers:
            return [],0,0
        next_offset = results["next"]
        papers = results["data"]
        #print(f'get article search returned first {len(papers)} papers of {total_papers}')
        for paper in papers:
            title = paper["title"]
            #print(f'considering {title}')
            if len(paper_library_df) > 0:
                dup = (paper_library_df['title'] == title).any()
                if dup:
                    print(f'*** {title} already indexed')
                    continue
            url = paper['url']
            isOpenAccess = paper['isOpenAccess']
            influentialCitationCount = int(paper['influentialCitationCount'])
            if isOpenAccess and 'openAccessPdf' in paper and type(paper['openAccessPdf']) is dict :
                openAccessPdf = paper['openAccessPdf']['url']
            else: openAccessPdf = None
            year = paper['year']
            if confirm:
                query = cot.confirmation_popup("Index this article?", title+"\n"+str(year))
                if not query:
                    print(f' skipping {title}')
                    continue
            abstract = paper['abstract'] if type(paper['abstract']) is str else str(paper['abstract'])
            authors = paper['authors']
            citationStyles = paper['citationStyles']
            citationCount = paper['citationCount']
            s2FieldsOfStudy = paper['s2FieldsOfStudy']
            tldr= paper['tldr']
            #embedding = paper['embedding']
            if abstract is None or (tldr is not None and len(tldr) > len(abstract)):
                abstract = str(tldr)
            if citationCount == 0:
                if year < 2020:
                    print(f'   skipping, no citations {title}')
                    continue
            result_dict = {key: '' for key in paper_library_columns}
            id = generate_faiss_id(lambda value: value in paper_library_df.faiss_id)
            result_dict["faiss_id"] = id
            result_dict["title"] = title
            result_dict["authors"] = [", ".join(author['name'] for author in authors)]
            result_dict["publisher"] = '' # tbd - double check if publisher is available from arxiv
            result_dict["summary"] = abstract 
            result_dict["citationCount"]= int(citationCount)
            result_dict["citationStyles"]= str(citationStyles)
            result_dict["inflCitations"] = int(influentialCitationCount)
            result_dict["evaluation"] = ''
            result_dict["pdf_url"] = openAccessPdf
            #print(f' url: {openAccessPdf}')
            pdf_filepath = None
            if openAccessPdf is not None:
                pdf_filepath= download_pdf(openAccessPdf, title)
            result_dict["pdf_filepath"]= pdf_filepath
            
            #print(f"indexing new article: {title}\n   pdf file: {type(result_dict['pdf_filepath'])}")
            result_dict['synopsis'] = ""
            result_dict['section_ids'] = [] # to be filled in after we get paper id
            #if status is None:
            #    continue
            #print(f' new article:\n{json.dumps(result_dict, indent=2)}')
            paper_index = len(paper_library_df)
            paper_library_df.loc[paper_index] = result_dict
            # section and index paper
            if not isOpenAccess:
                if (not influentialCitationCount > 0 and not confirm and
                    ((citationCount < 20 and year < 2020) or (citationCount <2 and year < 2021))):
                    continue
                if not confirm or not cot.confirmation_popup("try retrieving preprint?", query ):
                    continue
                preprint_url = get_arxiv_preprint_url(title)
                if preprint_url is None:
                    continue
                result_dict['pdf_url'] = preprint_url
                result_dict['pdf_filepath'] = download_pdf(preprint_url, title)
            queue_paper_for_indexing(result_dict)
            result_list.append(result_dict)
            
    except Exception as e:
        traceback.print_exc()
    #print(f'get articles returning')
    return result_list, total_papers, next_offset

# Test that the search is working
#result_output = get_articles("epigenetics in cancer")
#print(result_output[0])
#sys.exit(0)

def check_entities_against_draft(entities, draft):
    entities_in_text = []
    entities_not_in_text = []
    long_text=draft.lower()
    for entity in entities:
        if entity.lower() in long_text:
            entities_in_text.append(entity)
        else:
            entities_not_in_text.append(entity)
    return entities_in_text, entities_not_in_text


def index_paper_synopsis(paper_dict, synopsis, paper_index):
    global paper_indexIDMap
    if 'embedding' in paper_dict and paper_dict['embedding'] is not None:
        embedding = paper_dict['embedding']
    else:
        embedding = embedding_request(synopsis, 'search_document: ')
    ids_np = np.array([paper_dict['faiss_id']], dtype=np.int64)
    embeds_np = np.array([embedding], dtype=np.float32)
    paper_indexIDMap.add_with_ids(embeds_np, ids_np)
    paper_library_df.loc[paper_index, 'synopsis'] = synopsis
    return True

def index_section_synopsis(paper_dict, text):
    global section_indexIDMap
    paper_title = paper_dict['title']
    paper_authors = paper_dict['authors'] 
    paper_abstract = paper_dict['summary']
    pdf_filepath = paper_dict['pdf_filepath']

    text_embedding = embedding_request(text, 'search_document: ')
    faiss_id = generate_faiss_id(lambda value: value in section_library_df.index)
    ids_np = np.array([faiss_id], dtype=np.int64)
    embeds_np = np.array([text_embedding], dtype=np.float32)
    section_indexIDMap.add_with_ids(embeds_np, ids_np)

    ners = rw.section_entities(faiss_id, text, None)
    ner_embedding = embedding_request(', '.join(ners), 'search_document: ')
    ner_faiss_id = generate_faiss_id(lambda value: value in section_library_df.index)
    ner_ids_np = np.array([ner_faiss_id], dtype=np.int64)
    ner_embeds_np = np.array([ner_embedding], dtype=np.float32)
    section_indexIDMap.add_with_ids(ner_embeds_np, ner_ids_np)

    synopsis_dict = {"faiss_id":faiss_id,
                     "paper_id": paper_dict["faiss_id"],
                     "synopsis": text,
                     "ners": ners,
                     "ner_faiss_id": ner_faiss_id
                     }
    section_library_df.loc[len(section_library_df)]=synopsis_dict
    
    return faiss_id

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def index_paper(paper_dict):
    # main path for papers from index_url and / or index_file, where we know nothing other than pdf
    try:
        extract = grobid.parse_pdf(paper_dict['pdf_filepath'])
    except Exception as e:
        print(f'\ngrobid fail {str(e)}')
        traceback.print_exc()
        return False
    if extract is None or not extract:
        print('grobid extract is None')
        return False
    # print(f'index_paper grobid keys {extract.keys()}')
    title = None
    if 'title' in paper_dict and len(paper_dict['title']) >0:
        title = paper_dict['title']
    elif 'title' in extract and extract['title'] is not None:
        title = extract['title']
    else:
        title = paper_dict['pdf_filepath']
    print(f'title 1 {title}')
    paper_dict['authors'] = extract['authors']
    paper_dict['summary'] = extract['abstract'] # will be overwritten by S2 abstract if we can get it
    if title is not None:
        paper_dict['title']=title
        meta_data = get_title(title)
    if meta_data is not None:
        paper_dict["summary"] = str(meta_data['abstract'])
        #paper_dict["year"] = meta_data['year'])
        paper_dict["title"] = str(meta_data['title'])
        paper_dict["article_url"] = str(meta_data['url'])
        paper_dict["summary"] = str(meta_data['abstract'])
        paper_dict["citationCount"]= int(meta_data['citationCount'])
        paper_dict["citationStyles"]= str(meta_data['citationStyles'])
        paper_dict["inflCitations"] = int(meta_data['influentialCitationCount'])
    abstract = paper_dict['summary']
    if len(paper_library_df) > 0:
        dup = (paper_library_df['title'] == title).any()
        if dup:
            print(f' already indexed {title}')
            return False
    paper_faiss_id = generate_faiss_id(lambda value: value in paper_library_df.faiss_id)
    paper_dict['faiss_id'] = paper_faiss_id
    paper_index = len(paper_library_df)
    # paper_dict should be read only from here!
    if validate_paper(paper_dict):
        paper_library_df.loc[paper_index] = paper_dict
    text_chunks = [abstract]+extract['sections']
    section_synopses = []; section_ids = []
    rw_count = 0
    for idx, text_chunk in enumerate(text_chunks):
        if len(text_chunk) < 32:
            continue
        rw_count += 1
        sentences = [text_chunk]
        encoded_input = embedding_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        ### no more rewriting on ingest,
        ###   worried it distorts content more than content uniformity of style aids downstream extract
        
        id = index_section_synopsis(paper_dict, text_chunk)
        section_synopses.append(text_chunk)
        section_ids.append(id)
    return True

def sbar_as_text(sbar):
    return f"\n{sbar['needs']}\nBackground:\n{sbar['background']}\nReview:\n{sbar['observations']}\n"

def reverse_reciprocal_ranking(*lists):
    # Example usage
    #list1 = ["a", "b", "c"]
    #list2 = ["b", "c", "d"]
    #list3 = ["c", "d", "e"]
    #merged_list = reverse_reciprocal_ranking(list1, list2, list3)

    scores = {}
    for lst in lists:
        for i, item in enumerate(lst, start=1):
            if item not in scores:
                scores[item] = 0
            scores[item] += 1 / i

    # Create a merged list of items sorted by their scores in descending order
    merged_list = sorted(scores, key=scores.get, reverse=True)
    return merged_list


def hyde(query):
    hyde_prompt="Write a short sentence of 10-20 words responding to:\n{{$query}}\n\nEnd your sentence with: </RESPONSE>"
    prompt = [SystemMessage(hyde_prompt)]
    response = cot.llm.ask({"query":query},
                           prompt,
                           max_tokens=50,
                           temp=0.2,
                           eos='</RESPONSE>')
    end_idx =  response.lower().rfind('</RESPONSE>'.lower())
    if end_idx < 0:
        end_idx = len(response)+1
    return response[:end_idx]


def relevant (text, query, background):
    #print(f'enter relevant {query} {background}\n{text}')
    prompt = [SystemMessage("""Given the query:

<QUERY>
{{$query}}
{{$background}}
</QUERY>

<TEXT>
{{$text}}
</TEXT>

Respond using this JSON template. Return only JSON without any Markdown or code block formatting.

{"relevant": "Yes" or "No"}
""")
              ]
    response = cot.llm.ask({'text':text, 'query':query, 'background':background}, prompt, max_tokens=100, temp=0.1, stop_on_json=True, validator=JSONResponseValidator())
    #print(f"relevant {response['relevant']}\n")
    if type(response) == dict:
        if 'relevant' in response and 'yes' in str(response['relevant']).lower():
            return True
    return False

def search(query, dscp='', top_k=20, web=False, whole_papers=False, query_is_title=False):
    
    """Query is a *list* of keywords or phrases
    - 'dscp' is an expansion that can be used for reranking using llm
    - if query_is_title then whole_papers == True!
    - Finds the closest n faiss IDs to the user's query
    - returns the section synopses and section_library and paper_library entries for the found items
        -- or just one section for each paper if whole_papers=True"""
    
    # A prompt to dictate how the recursive summarizations should approach the input paper
    #get_ articles does arxiv library search. This should be restructured
    results = []
    i = 0
    next_offset = 0
    total = 999
    #query = cot.confirmation_popup("Search SemanticScholar using this query?", query )
    while web and next_offset < total:
        results, total, next_offset = get_articles(query, next_offset)
        i+=1
        print(f"arxiv search returned {len(results)} papers")
        for paper in results:
            print(f"    title: {paper['title']}")

    # arxiv search over, now search faiss
    rw.cot = cot #just to be sure...
    if dscp is not None and len(dscp) > 0:
        top_k = top_k * 1.5 # get more so we can rerank
        rerank = True

    query_ids, query_summaries = search_sections(query+'. '+dscp, top_k=int(top_k))
    #print(f'search ids {query_ids}')
    kwd_ids, kwd_summaries = search_sections(' '.join(rw.extract_entities(query+'. '+dscp)), top_k=int(top_k))
    #print(f'kwd ids {kwd_ids}')
    hyde_query = hyde(query+'. '+dscp)
    #print(f'hyde query: {hyde_query}')
    hyde_ids, hyde_summaries = search_sections(hyde_query, top_k=int(top_k))
    #print(f'hyde ids {hyde_ids}')
    if query_is_title:
        # use scikit Tfidf on titles
        vectorizer = TfidfVectorizer()
        titles = paper_library_df['title'].tolist()
        tfidf_matrix = vectorizer.fit_transform(titles + [query])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        tdidf_ids = []
        for loc, value in enumerate(cosine_similarities[0]):
            if value > 0.0:
                print(loc, paper_library_df.iloc[loc]['title'])
            tdidf_ids.append((value, paper_library_df.iloc[loc]['faiss_id']))
        #print(f'title similarities {query}\n{cosine_similarities}')
        sorted_tuples = sorted(tdidf_ids, key=lambda x: x[0], reverse=True)
        sorted_tdidf_ids = [item[1] for item in sorted_tuples]
        print(f'tdidfs {sorted_tdidf_ids}')

        id_mapping = dict(zip(section_library_df['faiss_id'], section_library_df['paper_id']))
        query_ids_mapped = [id_mapping[query_id] for query_id in query_ids if query_id in id_mapping]
        kwd_ids_mapped = [id_mapping[kwd_id] for kwd_id in kwd_ids if kwd_id in id_mapping]
        hyde_ids_mapped = [id_mapping[hyde_id] for hyde_id in hyde_ids if hyde_id in id_mapping]
        print(f'query ppr ids {query_ids_mapped}')
        print(f'kwd ppr ids {kwd_ids_mapped}')
        print(f'hyde ppr ids {hyde_ids_mapped}')

        reranked_ids = reverse_reciprocal_ranking(query_ids_mapped, kwd_ids_mapped, hyde_ids_mapped, sorted_tdidf_ids[:int(top_k)])[:int(top_k)]
        print(f'reranked paper ids {reranked_ids}')
        return reranked_ids, []
    else:
        reranked_ids = reverse_reciprocal_ranking(query_ids, kwd_ids, hyde_ids)[:int(top_k)]
        excerpts = []
        for section_id in reranked_ids:
            if section_id in query_ids:
                index = query_ids.index(section_id)
                excerpts.append(query_summaries[index])
            elif section_id in kwd_ids:
                index = kwd_ids.index(section_id)
                excerpts.append(kwd_summaries[index])
            elif section_id in hyde_ids:
                index = hyde_ids.index(section_id)
                excerpts.append(hyde_summaries[index])
                
    return reranked_ids, excerpts

def parse_arguments():
    """Parses command-line arguments using the argparse library."""

    parser = argparse.ArgumentParser(description="Process a single command-line argument.")

    parser.add_argument("-template", type=str, help="the LLM template to use")
    parser.add_argument("-index_url", type=str, help="index a pdf url")
    parser.add_argument("-index_file", type=str, help="index a pdf from filepath")
    parser.add_argument("-index_paper", type=str, help="index a paper - internal use only")
    parser.add_argument("-browse", type=str, help="browse local library")
    args = parser.parse_args()
    return args

def queue_paper_for_indexing(paper_dict):
    print(f'queue_paper {str(paper_dict)}')
    response = requests.post(f'http://127.0.0.1:5006/submit_paper/', params={'paper':json.dumps(paper_dict)})
    data = response.json()
    print(f'paper submitted for indexing, response: {data}')
    return data

def queue_url_for_indexing(url):
    response = requests.post(f'http://127.0.0.1:5006/submit_url/', params={'url':url})
    data = response.json()
    print(f'url submitted for indexing, response: {data}')
    return data

def repair_paper_library_from_section_library():
    pass

class PaperSelect(QWidget):
    def __init__(self, papers, search_terms):
        super().__init__()
        self.papers = papers
        self.dscp = search_terms
        self.title_rows = []
        self.select_rows = []
        self.show_rows = []
        layout = QVBoxLayout()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#444444"))  # Use any hex color code
        palette.setColor(self.foregroundRole(), QtGui.QColor("#EEEEEE"))  # Use any hex color code
        self.setPalette(palette)
        self.codec = QTextCodec.codecForName("UTF-8")
        self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
        for paper in self.papers:
            row_layout = QHBoxLayout()
            select = QCheckBox(paper, self)
            row_layout.addWidget(select)
            show = QPushButton('Show', self)
            show.clicked.connect(lambda _, p=paper: self.show_paper(p))
            row_layout.addWidget(show)
            self.title_rows.append(paper)
            self.select_rows.append(select)
            self.show_rows.append(show)
            layout.addLayout(row_layout)

        # Search Button
        discuss_button = QPushButton('Discuss', self)
        discuss_button.clicked.connect(self.on_discuss_clicked)
        layout.addWidget(discuss_button)
        show_button = QPushButton('Show', self)
        show_button.clicked.connect(self.on_show_clicked)
        layout.addWidget(show_button)

        self.setLayout(layout)

        self.setWindowTitle('Papers')
        self.setGeometry(300, 300, 400, 250)
        self.show()

    def show_paper(self, paper):
        print(f'show paper {paper}')
        search_result = paper_library_df[paper_library_df['title'].str.contains(paper, na=False)]
        if search_result is not None and len(search_result) > 0:
            # first display orig ppr
            filepath = str(search_result.iloc[0]["pdf_filepath"])
            if filepath is not None and len(filepath) > 0:
                print(f'filepath: {filepath}')
                if filepath.startswith('/home'):
                    pass
                elif filepath.startswith('./arxiv'):
                    filepath = '/home/bruce/Downloads/owl/tests/owl'+filepath[1:]
                elif filepath.startswith('/arxiv'):
                    filepath = '/home/bruce/Downloads/owl/tests/owl'+filepath
                elif filepath.startswith('arxiv'):
                    filepath = '/home/bruce/Downloads/owl/tests/owl/'+filepath
                uri= 'file://'+filepath # assumes filepath is an absolute path starting /home
                if not Path(uri).exists():
                    uri = str(search_result.iloc[0]["pdf_url"])
            if uri is not None and len(uri) > 0:
                webbrowser.open_new(uri)
            #now display excerpt sections
            paper_id = str(search_result.iloc[0]["faiss_id"])
            print(f'paper_id {paper_id}')
            sections = section_library_df[section_library_df['paper_id'].astype(str) == str(paper_id)]
            if sections is not None and len(sections) > 0:
                for s, section in sections.iterrows():
                    print(f'\nSection {s}')
                    print(f"{section['synopsis']}")
            else:
                print(f'\nNo sections found for paper\n')
        
    def collect_checked_resources(self):
        # 'resources' must match return from search, that is: [[section_id, ...], [[paper_title, section_excerpt],...]
        resources = []
        section_ids = [] # a list of all section ids (in this case, all for every paper selected!)
        excerpts = [] # a list of [paper_title, section_synopsis] for every section of every paper
        for title_row, select_row, show_row in zip(self.title_rows, self.select_rows, self.show_rows):
            if not select_row.isChecked():
                continue
            title=title_row
            search_result = paper_library_df[paper_library_df['title'].str.contains(title, na=False)]
            if len(search_result) > 0: # found paper, gather all sections
                paper_id = str(search_result.iloc[0]["faiss_id"])
                sections = section_library_df[section_library_df['paper_id'].astype(str) == str(paper_id)]
                print(f'sections in paper {paper_id}, {len(sections)}')
                paper_sections = sections['faiss_id'].tolist()
                paper_excerpts = [[title, synopsis] for synopsis in sections['synopsis'].tolist()]
                section_ids.extend(paper_sections)
                excerpts.extend(paper_excerpts)
        return [section_ids, excerpts]
    
    def first_checked_paper_url(self):
        # returns the pdf url (file or http) for the first checked paper
        for title_row, select_row in zip(self.title_rows, self.select_rows):
            if not select_row.isChecked():
                continue
            title=title_row
            search_result = paper_library_df[paper_library_df['title'].str.contains(title, na=False)]
            if len(search_result) > 0:
                paper_id = str(search_result.iloc[0]["faiss_id"])
                filepath = str(search_result.iloc[0]["pdf_filepath"])
                if filepath is not None and len(filepath) > 0:
                    return 'file:///home/bruce/Downloads/owl/tests/owl/'+filepath
                http = str(search_result.iloc[0]["pdf_url"])
                if url is not None and len(url) > 0:
                    return http
        return None

    def on_discuss_clicked(self):
        """ Handles the discuss button click event. """
        resources = {}
        resources['sections'] = self.collect_checked_resources()
        resources['dscp'] = self.dscp
        resources['template'] = cot.template
        with open('discuss_resources.pkl', 'wb') as f:
            pickle.dump(resources, f)
        rc = subprocess.run(['python3', 'paper_writer.py', '-discuss', 'discuss_resources.pkl'])
        
    def on_show_clicked(self):
        """ Handles the show button click event. """
        pdf_url = self.first_checked_paper_url()
        if pdf_url is not None:
            webbrowser.open_new(pdf_url)
        else:
            print(f"Sorry, I don't have file or url")


class BrowseUI(QWidget):
    def __init__(self, papers):
        super().__init__()
        self.parent_papers = papers # a dict to receive result
        layout = QVBoxLayout()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#444444"))  # Use any hex color code
        palette.setColor(self.foregroundRole(), QtGui.QColor("#EEEEEE"))  # Use any hex color code
        self.setPalette(palette)
        self.codec = QTextCodec.codecForName("UTF-8")
        self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
        self.rows = {'Title':None, 'Abstract':None, 'Year':None}
        for row in self.rows.keys():
            row_layout = QHBoxLayout()
            label = QLabel(row)
            row_layout.addWidget(label)
            self.rows[row] = QLineEdit(row)
            row_layout.addWidget(self.rows[row])
            layout.addLayout(row_layout)

        # Search Button
        search_button = QPushButton('Search', self)
        search_button.clicked.connect(self.on_search_clicked)
        layout.addWidget(search_button)

        self.setLayout(layout)

        self.setWindowTitle('Browse Library')
        self.setGeometry(300, 300, 400, 250)
        self.show()

    def on_search_clicked(self):
        """
        Handles the search button click event. 
        """
        search_config={}
        for row_label in self.rows.keys():
             search_config[row_label] = self.rows[row_label].text()
        print(search_config)
        query = ''; dscp = ''
        if search_config['Title'] is not None  and search_config['Title'] != 'Title':
            query = search_config['Title'].strip()
        if search_config['Abstract'] is not None  and search_config['Abstract'] != 'Abstract':
            dscp = search_config['Abstract'].strip()
        if query == '' and dscp == '':
            print(f'Please enter title and/or abstract')
            return
        elif query == '': # use abstract search terms for primary search
            query = dscp

        # browse is about papers, not sections
        # 'whole_papers=True' only returns one section for each paper
        finds = search(query, dscp, whole_papers=True, query_is_title=True) 
        # finds is a list: [[section_id,...], [[paper_title, section_text],...]]
        paper_ids = finds[0]
        excerpts = finds[1] # so note excerpts is a list of [paper_title, section_excerpt]
        #
        ### filter out duplicates at paper level and display list of found titles.
        #
        filepaths = []
        paper_titles = []
        for paper_id in paper_ids:
            search_result = paper_library_df[paper_library_df['faiss_id']==paper_id]
            if len(search_result) > 0:
                paper_titles.append(search_result.iloc[0]['title'])
                filepath = search_result.iloc[0]["pdf_filepath"]
                print(f'filepath: {filepath}')
                if filepath.startswith('/home'):
                    pass
                elif filepath.startswith('./arxiv'):
                    filepath = '/home/bruce/Downloads/owl/tests/owl'+filepath[1:]
                elif filepath.startswith('/arxiv'):
                    filepath = '/home/bruce/Downloads/owl/tests/owl'+filepath[0:]
                elif filepath.startswith('arxiv'):
                    filepath = '/home/bruce/Downloads/owl/tests/owl/'+filepath
                #pubdate = search_result.iloc[0]["year"]
                filepaths.append(filepath)
                #print(f"   Found file: {filepath}")
            else:
                filepaths.append(None)
                #print("    No matching rows found.")
        self.parent_papers['titles'] = paper_titles
        self.parent_papers['filepaths'] = filepaths
        self.parent_papers['years'] = filepaths
        self.parent_papers['dscp'] = dscp
        self.close()

    
def browse():
    global updated_json, config
    papers = {}
    ex = BrowseUI(papers) # get config for this discussion task
    ex.show()
    app.exec()
    # display chooser, then call discuss with selected pprs (or excerpts?)
    chooser = PaperSelect(papers['titles'], papers['dscp']) # get config for this discussion task
    chooser.show()
    app.exec()

    #get updated config

if __name__ == '__main__':
    import OwlCoT as cot
    args = parse_arguments()
    template = None
    if hasattr(args, 'template') and args.template is not None:
        template = args.template
        print(f' S2 using template {template}')
    cot = cot.OwlInnerVoice(None)
    rw.cot = cot
    if hasattr(args, 'index_url') and args.index_url is not None:
        url = args.index_url.strip()
        print(f' S2 indexing url {url}')
        index_url(url)
    if hasattr(args, 'index_file') and args.index_url is not None:
        filepath = args.index_file.strip()
        print(f' S2 indexing file {filepath}')
        index_file(filepath)
    if hasattr(args, 'index_paper') and args.index_paper is not None:
        try:
            paper = json.loads(args.index_paper.strip())
            print(f' S2 indexing ppr {ppr}')
            index_paper(paper)
        except Exception as e:
            print(str(e))
    if hasattr(args, 'browse') and args.browse == 't':
        app = QApplication(sys.argv)
        browse()
        
    else:
        print(' -index_url, -index_file, or -index_paper')
    """
        papers_dir = "/home/bruce/Downloads/owl/tests/owl/arxiv/papers/"

        papers = [d for d in os.listdir(papers_dir) ]
        print(f'papers: {len(papers)}')
        for p, paper in enumerate(papers):
            #if p > 10:
            #    break
            print(paper)
            index_url("file://"+papers_dir+paper)

        """
