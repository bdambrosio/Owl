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
import semanticScholar3 as s2
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


paper_library_columns = ["faiss_id", "title", "authors", "citationStyles", "publisher", "summary", "inflCitations", "citationCount", "article_url", "pdf_url","pdf_filepath", "section_ids"]

paper_library_df = pd.read_parquet(paper_library_filepath)
print('loaded paper_library_df')
    
paper_indexIDMap = faiss.read_index(paper_index_filepath)
print(f"loaded '{paper_index_filepath}'")
    
section_indexIDMap = faiss.read_index(section_index_filepath)
print(f"loaded '{section_index_filepath}'")
    
section_library_df = pd.read_parquet(section_library_filepath)
print(f"loaded '{section_library_filepath}'\n  keys: {section_library_df.keys()}")

def parse_arguments():
    """Parses command-line arguments using the argparse library."""

    parser = argparse.ArgumentParser(description="Process a single command-line argument.")

    parser.add_argument("-template", type=str, help="the LLM template to use")
    parser.add_argument("-search", type=str, help="search for term and descendants")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import OwlCoT as cot
    args = parse_arguments()
    template = None
    if hasattr(args, 'template') and args.template is not None:
        template = args.template
        print(f' S2 using template {template}')
        cot = cot.OwlInnerVoice(None, template=template)
    else:
        cot = cot.OwlInnerVoice(None)
    s2.cot = cot
    rw.cot = cot
    if hasattr(args, 'search') and args.search is not None:
        search_term = args.search
        print(f' l2os searching {search_term}')
        query = search_term.strip()
        ids, summaries = s2.search(query, '')
        print(ids)
        titles = [item[0] for item in summaries]
        print(titles[0])
        #titles = list(set(titles))
        texts = [item[1] for item in summaries]
        prompt = [SystemMessage("""User will provide an NER and ask you to select items from a list of candidate NERs that are either subclasses or members of the provided NER. Respond only with the list of selected candidates, ending with:
</END>

For example:

NER: 
miRNA

Candidates:
protein
let-7
mRNA
gene
miRNA-223
hsa-let-7a
lncRNA
gene expression regulator

Response:
let-7
miRNA-223
hsa-let-7a
</END>"""),
                  UserMessage("NER:\nmolecule\n\nCandidates:\nNaCl\nSodium\nPCl\nglue"),
                  AssistantMessage('Response:\n')]
        
        response = cot.llm.ask('', prompt)
        print(response)        

        prompt = [SystemMessage("""User will provide an NER and ask you to provide a list of NERs that are generalizations (superclasses) of the provided NER. Respond only with the list of superclasses, ending with:
</END>

For example:

From the perspective of a detailed biomolecular ontology, list generalizations (eg, superclasses) of the NER: miRNA

Superclasses:

Nucleic Acids
RNA
Ribonucleic Acid
ncRNA
Non-coding RNA
Small RNAs
Regulatory RNA
Gene Expression Regulators
</END/>
"""
                        ),
                  UserMessage('NER:\n small RNA'),
                  AssistantMessage('Response:')]
        
        response = cot.llm.ask('', prompt)
        print(response)        
    else:
        print(' -search')
