import os, sys
import pandas as pd
import arxiv
from arxiv import Client, Search, SortCriterion, SortOrder
import ast
import concurrent
from csv import writer
from IPython.display import display, Markdown, Latex
import json
import re
import openai
import traceback
import pdfminer.high_level as miner
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import requests
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
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
from SamCoT import LLM, ListDialog
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from Planner import Planner
from transformers import AutoTokenizer, AutoModel

# startup AI resources

# load embedding model and tokenizer
embedding_tokenizer = AutoTokenizer.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')
#load base model
embedding_model = AutoModel.from_pretrained('/home/bruce/Downloads/models/Specter-2-base')
#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
embedding_model.load_adapter('/home/bruce/Downloads/models/Specter-2', source="hf", load_as="proximity", set_active=True)

import Sam as sam
ui = sam.window
ui.reflect=False # don't execute reflection loop, we're just using the UI
cot = ui.samCoT
pl = Planner(ui, cot)
 

OPENAI_MODEL3 = "gpt-3.5-turbo-16k"
OPENAI_MODEL4 = "gpt-4-1106-preview"
#OPENAI_MODEL = "gpt-4-32k"
#EMBEDDING_MODEL = "text-embedding-ada-002"
OS_MODEL='zephyr'

openai_api_key = os.getenv("OPENAI_API_KEY")
openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
llm = LLM(None, None, osClient=OSClient(api_key=None), openAIClient=openAIClient, template=OS_MODEL)

directory = './data/papers/'

# Check if the directory already exists
# Set a directory to store downloaded papers
data_dir = os.path.join(os.curdir, "data", "papers")
paper_dir_filepath = "./data/arxiv_library.parquet"
plans_filepath = "./data/arxiv_plans.json"
if not os.path.exists(directory):
    # If the directory doesn't exist, create it and any necessary intermediate directories
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    # If the directory already exists, print a message indicating it
    print(f"Directory '{directory}' exists.")
if not os.path.exists(paper_dir_filepath):
    # Generate a blank dataframe where we can store downloaded files
    df = pd.DataFrame(columns=["title","authors","summary","article_url", "pdf_url","pdf_filepath", "embedding"])
    df.to_parquet(paper_dir_filepath)
    print('df.to_parquet initialization complete')

plans = {}
if os.path.exists(plans_filepath):
    with open(plans_filepath, 'r') as f:
        plans = json.load(f)
        print(f'loaded plans.json')
else:
    print(f'initializing plans.json')
    plans = {}


#@retry(wait=wait_random_exponential(min=2, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    text_batch = [text]
    # preprocess the input
    inputs = embedding_tokenizer(text_batch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = embedding_model(**inputs)
    # take the first token in the batch as the embedding
    embedding = output.last_hidden_state[0, 0, :]
    print(f'embedding_request response shape{embedding.shape}')
    return embedding.detach().numpy()

# Function to recursively extract string values from nested JSON
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

# Function to extract words from JSON
def extract_words_from_json(json_data):
    string_values = extract_string_values(json_data)
    text = ' '.join(string_values)
    words = re.findall(r'\b\w+\b', text)
    return words

def extract_keywords(text):
    keywords = []
    for word in text:
        zipf = wf.zipf_frequency(word, 'en', wordlist='large')
        print(word, zipf)
        if zipf < 3.5 and word not in keywords:
            keywords.append(word)
    return keywords

def search(search_anyway=False):
    global plans
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
                #create json file of available plans - why here?
                with open(plans_filepath, 'w') as f:
                    json.dump(plans, f)
    else:
        # new plan, didn't select any existing
        plan = pl.init_plan()
        plan = pl.analyze()
        print(json.dumps(plan, indent=4))
    # at this point pl.active_plan looks like:
    #   {"name":'plannnn_aaa', "dscp":'bbb', "sbar":{"needs":'q\na', "background":'q\na',"observations":'q\na'}

    if 'arxiv_keywords' not in plan:
        # extract a list of keywords from the title and sbar
        # Extracting words
        extracted_words = extract_words_from_json(plan['sbar'])
        keywords = extract_keywords(extracted_words)
        print(keywords)
        plan['arxiv_keywords'] = keywords
    else:
        keywords = plan['arxiv_keywords']
    summarize_text(keywords, search_anyway=search_anyway)

    
def get_articles(query, library_file=paper_dir_filepath, top_k=20):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in arxiv_library.csv to be retrieved by the read_article_and_summarize.
    """

    # Initialize a client
    result_list = []
    library_df = pd.read_parquet(library_file).reset_index()
    print(f'get_articles query: {query}, {type(library_df)}')
    try:
        # Set up your search query
        search = arxiv.Search(
            query=query, 
            max_results=top_k, 
            sort_by=SortCriterion.Relevance,
            sort_order = SortOrder.Descending
        )
    
        print(f'get article search complete {search}')
        # Use the client to get the results
        for result in arxiv.Client().results(search):
            if len(library_df) > 0:
                dup = (library_df['title']== result.title).any()
                if dup:
                    # skip articles already in lilbrary_df
                    continue
            result_dict = {"title":'',"authors":'',"summary":'',"article_url":'', "pdf_url":'', "pdf_filepath":'', "embedding":''}
            result_dict["title"] = result.title
            result_dict["authors"] = author_names = [", ".join(author.name for author in result.authors)]
            result_dict["summary"] = result.summary
            # Taking the first url provided
            result_dict["article_url"] =  [x.href for x in result.links][0]
            result_dict["pdf_url"] = [x.href for x in result.links][1]
            result_dict["pdf_filepath"]=result.download_pdf(data_dir),
            embedding = embedding_request(text=result.title+'\n'+result.summary)
            result_dict["embedding"] =  embedding
            #print([x.href for x in result.links][1])
            print(f'get_articles title {result.title}')
            library_df = library_df._append(result_dict, ignore_index=True)

        # Write to file
        library_df.to_parquet(library_file)
    except Exception as e:
        traceback.print_exc()
    #print(f'get articles returning')
    return result_list

# Test that the search is working
#result_output = get_articles("epigenetics in cancer")
#print(result_output[0])
#sys.exit(0)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding = embedding_request(text=str(query))
    strings_and_relatednesses = [
        (row["pdf_filepath"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    print(strings[0])
    return strings[:top_n]

def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    #reader = PdfReader(filepath)
    #pdf_text = ""
    #page_number = 0
    #for page in reader.pages:
    #    page_number += 1
    #    pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    print(f'pdf filepath {filepath}')
    pdf_text = miner.extract_text(filepath)
    print(f'pdf len {len(pdf_text)}')
    return pdf_text


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(pdf_text):
    messages=[UserMessage(f"""Extract the title, authors, and outline from the following pdf text.
For each outline section identified, report the section title.
Using the following JSON format:
{{"title": '<pdf title>',
 "authors": '<authors>',
 "sections": ['<section title>', '<section title>',...]
}}
                        PDF text:\n{pdf_text[:25000]}
""")
                  ]
    response = llm.ask('', messages, client=openAIClient, template=OPENAI_MODEL4, max_tokens = 600,temp=0.0, validator = JSONResponseValidator())
    print(response)
    
    if type(response) is dict and 'sections' in response:
       chunks = []
       start=0 # can't start before start of prev section
       for s, section_title in enumerate(response['sections']):
          # retrieve text of section - we could just try find in text, maybe?
          begin = (pdf_text[start:]).find(section_title)
          if start > -1:
             if s < len(response['sections']) - 2:
                end = pdf_text[start+begin:].find(response['sections'][s+1])
             else:
                end = len(pdf_text) - start - begin
             if end > -1:
                section_text = pdf_text[start+begin:start+begin+end]
                start += begin
                print(f'{section_title}, length: {end}')
                chunks.append(f'{section_title}\n{pdf_text[start:start+end]}')
       return chunks
   

def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarized chunk of text"""
    prompt = [UserMessage(template_prompt+content),
              AssistantMessage('')
              ]
    response = llm.ask('', prompt, client=openAIClient, template=OPENAI_MODEL3, temp=0.0)
    print(f'extract_chunk {response}')
    return response

def extract_toc(filename, maxlevel):
    with open(filename, 'rb') as fp:
        parser = PDFParser(fp)
        doc = PDFDocument(parser)
        outlines = doc.get_outlines()
        print(f'\n\noutlines:')
        for (level, title, dest, a, se) in outlines:
            #if level <= maxlevel:
            print(level, title, dest, a, se)
    print(f'\n\n')

def summarize_text(query, search_anyway=False):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. The summary should include the central argument of the text, together with all key points supporting that central argument. A key point might be any observation, statement, fact, inference, question, hypothesis, conclusion, or decision relevant to the central argument of the text. \n\nContent:"""

    # If the library is empty (no searches have been performed yet), we perform one and download the results
    library_df = pd.read_parquet(paper_dir_filepath).reset_index()
    print("summarize text entry, num pprs found: ",len(library_df))
    if len(library_df) == 0 or search_anyway:
        print("No papers searched yet or search_anyway, downloading first.")
        get_articles(', '.join(query[0:9]))
        print("Papers downloaded, continuing")
        library_df = pd.read_parquet(paper_dir_filepath).reset_index()
    #library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    pdf_text = read_pdf(strings[0][0])
    
    #extract_toc(strings[0],2) # doesn't return anything, maybe bad code
    
    # Chunk up the document by section
    text_chunks = create_chunks(pdf_text)
    print(f"Summarizing each chunk of text {len(text_chunks)}")
    results = ''
    # Parallel process the summaries
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        futures = [
            executor.submit(extract_chunk, chunk, summary_prompt)
            for chunk in text_chunks
        ]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            if data is not None:
                results += data

    # Final summary
    print(f'\n\nChunk summaries\n{results}\n\n')
    print("Summarizing into overall summary")
    
    messages=[UserMessage(f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""")
                  ]
    response = llm.ask('', messages, temp=0.0)
    print(response)

class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )


# Start with a system message
paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions.
You summarize the papers clearly so the customer can decide which to read to answer their question.
You always provide the article_url and title so the user can understand the name of the paper and click through to access it.
Begin!"""
#paper_conversation = Conversation()
#paper_conversation.add_message("system", paper_system_message)

# Test that the search is working
#result_output = get_articles("ppo reinforcement learning")
#print(result_output[0])


#chat_test_response = summarize_text("PPO reinforcement learning sequence generation")
if __name__ == '__main__':
    #get_articles('PPO Optimization')
    #print(summarize_text('Prediction of tissue-of-origin of early stage cancers using serum miRNomes'))
    #print(summarize_text('Gene Expression Microarray Analysis'))
    #search('Gene Expression Microarray Analysis')
    search(search_anyway=False)
