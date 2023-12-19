import os, sys, logging
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
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
from OwlCoT import LLM, ListDialog, generate_faiss_id
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from Planner import Planner
from transformers import AutoTokenizer, AutoModel

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


import Owl as owl
ui = owl.window
ui.reflect=False # don't execute reflection loop, we're just using the UI
cot = ui.owlCoT
pl = Planner(ui, cot)
 

OPENAI_MODEL3 = "gpt-3.5-turbo-16k"
OPENAI_MODEL4 = "gpt-4-1106-preview"
#OPENAI_MODEL = "gpt-4-32k"
#EMBEDDING_MODEL = "text-embedding-ada-002"
ssKey = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
OS_MODEL='zephyr'

openai_api_key = os.getenv("OPENAI_API_KEY")
openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
memory = VolatileMemory()
llm = LLM(None, memory, osClient=OSClient(api_key=None), openAIClient=openAIClient, template=OS_MODEL)
logging.basicConfig(level=logging.DEBUG)
directory = './arxiv/'
# Set a directory to store downloaded papers
papers_dir = os.path.join(os.curdir, "arxiv", "papers")
paper_library_filepath = "./arxiv/paper_library.parquet"
plans_filepath = "./arxiv/arxiv_plans.json"
plans = {}

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
        

def get_semantic_scholar_meta(doi):
    url = f"https://api.semanticscholar.org/v1/paper/doi:{doi}"
    #url = f"https://api.semanticscholar.org/v1/paper/{s2_id}"
    headers = {'x-api-key':ssKey}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        #print(data.keys())
        citations = data.get("citationVelocity")
        influential = data.get("influentialCitationCount")
        publisher = data.get("publicationVenue")
        return citations, influential, publisher
    else:
        print("Failed to retrieve data for arXiv ID:", arxiv_id)
        return 0,0,''

def paper_library_df_fixup():
    return

paper_library_columns = ["faiss_id", "title", "authors", "publisher", "summary", "inflCitations", "citationCount", "evaluation", "article_url", "pdf_url","pdf_filepath", "synopsis", "section_ids"]

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
    #paper_library_df_fixup()
    print('loaded paper_library_df')
    
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
        pd.DataFrame(columns=["faiss_id", "paper_id", "synopsis"])
    section_library_df.to_parquet(section_library_filepath)
    print(f"created '{section_library_filepath}'")
else:
    section_library_df = pd.read_parquet(section_library_filepath)
    print(f"loaded '{section_library_filepath}'\n  keys: {section_library_df.keys()}")

def save_synopsis_data():
    faiss.write_index(paper_indexIDMap, paper_index_filepath)
    faiss.write_index(section_indexIDMap, section_index_filepath)
    section_library_df.to_parquet(section_library_filepath)
    
def search_sections(query, sbar, top_k=20):
    #print(section_library_df)
    # get embed
    expanded_query = " ".join(query)+sbar_as_text(sbar)
    query_embed = embedding_request(expanded_query)
    # faiss_search
    embeds_np = np.array([query_embed], dtype=np.float32)
    scores, ids = section_indexIDMap.search(embeds_np, top_k)
    print(f'ss ids {ids}, scores {scores}')
    # lookup text in section_library
    sections = []
    for id in ids[0]:
        section_library_row = section_library_df[section_library_df['faiss_id'] == id]
        #print section text
        if len(section_library_row) > 0:
            row = section_library_row.iloc[0]
            text = row['synopsis']
            article_title = row['title']
            synopses.append([article_title, text])
            print(f'{article_title} {len(text)}')
    return synopses

    
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
        if zipf < 3.75 and word not in keywords:
            keywords.append(word)
    return keywords

def plan_search(web=False):
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
        plan = pl.analyze()
        # store new plan in list of search plans
        plans[plan['name']] = plan
        with open(plans_filepath, 'w') as f:
            json.dump(plans,f)
        print(json.dumps(plan, indent=4))
        #   {"name":'plannnn_aaa', "dscp":'bbb', "sbar":{"needs":'q\na', "background":'q\na',"observations":'q\na'}

    if plan is None:
        return
    if 'arxiv_keywords' not in plan:
        # extract a list of keywords from the title and sbar
        # Extracting words
        extracted_words = extract_words_from_json(plan['sbar'])
        keywords = extract_keywords(extracted_words)
        plan['arxiv_keywords'] = keywords
    else:
        keywords = plan['arxiv_keywords']
    search(keywords,plan['sbar'],  web=web)

def get_articles(query, library_file=paper_library_filepath, top_k=100):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in paper_library.csv to be retrieved by the read_article_and_summarize.
    """

    # Initialize a client
    result_list = []
    #library_df = pd.read_parquet(library_file).reset_index()
    query = ' and '.join(query)
    print(f'get_articles query: {query}')
    query = cot.confirmation_popup("Search ARXIV using this query?", query )
    if not query:
        print(f'confirmation: No!')
        return []

    try:
        # Set up your search query
        query='Direct Preference Optimization for large language model fine tuning'
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=url,title,year,abstract,authors,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,s2FieldsOfStudy,tldr"
        headers = {'x-api-key':ssKey, }
        
        response = requests.get(url, headers = headers)
        if response.status_code != 200:
            print(f'SemanticsSearch fail code {response.status_code}')
            return []
        results = response.json()
        total_papers = results["total"]
        current_offset = results["offset"]
        next_offser = results["next"]
        papers = results["data"]
        print(f'get article search returned first {len(papers)} papers of {total_papers}')
        for paper in papers:
            title = paper["title"]
            print(f'considering {title}')
            if len(paper_library_df) > 0:
                dup = (paper_library_df['title'] == title).any()
                if dup:
                    print(f'   already indexed {title}')
                    continue
            url = paper['url']
            isOpenAccess = paper['isOpenAccess']
            influentialCitationCount = paper['influentialCitationCount']
            if not isOpenAccess:
                if influentialCitationCount > 0:
                    print(f'   not open access but influentialCitationCount {influentialCitationCount}')
                    continue
            openAccessPdf = paper['openAccessPdf']
            year = paper['year']
            abstract = paper['abstract']
            authors = paper['authors']
            citationCount = paper['citationCount']
            openAccessPdf = paper['openAccessPdf']
            s2FieldsOfStudy = paper['s2FieldsOfStudy']
            tldr= paper['tldr']
            if abstract is None or (tldr is not None and len(tldr) > len(abstract)):
                abstract = tldr
            if citationCount == 0:
                if year < 2022:
                    print(f'   skipping, no citations {result.title}')
                    continue
            result_dict = {key: '' for key in paper_library_columns}
            id = generate_faiss_id(lambda value: value in paper_library_df.faiss_id)
            result_dict["faiss_id"] = id
            result_dict["title"] = title
            result_dict["authors"] = [", ".join(author['name'] for author in authors)]
            result_dict["publisher"] = '' # tbd - double check if publisher is available from arxiv
            result_dict["summary"] = abstract
            result_dict["citationCount"]= citationCount
            result_dict["inflCitations"] = influentialCitationCount
            result_dict["evaluation"] = ''
            result_dict["pdf_url"] = openAccessPdf
            pdf_filepath= download_pdf(openAccessPdf, dir=papers_dir)
            result_dict["pdf_filepath"]= pdf_filepath
            
            print(f"indexing new article: {result.title}\n   pdf file: {type(result_dict['pdf_filepath'])}")
            result_dict['synopsis'] = ""
            result_dict['section_ids'] = [] # to be filled in after we get paper id
            #if status is None:
            #    continue
            print(f' new article:\n{json.dumps(result_dict, indent=2)}')
            paper_index = len(paper_library_df)
            paper_library_df.loc[paper_index] = result_dict
            # section and index paper
            paper_synopsis, paper_id, section_synopses, section_ids = index_paper(result_dict)
            paper_library_df.to_parquet(library_file)
            result_list.append(result_dict)
            
    except Exception as e:
        traceback.print_exc()
    #print(f'get articles returning')
    # get_articles assumes someone downstream parses and indexes the pdfs
    return result_list

# Test that the search is working
#result_output = get_articles("epigenetics in cancer")
#print(result_output[0])
#sys.exit(0)

def index_paper_synopsis(paper_dict, synopsis):
    global paper_indexIDMap
    embedding = embedding_request(synopsis)
    ids_np = np.array([paper_dict['faiss_id']], dtype=np.int64)
    embeds_np = np.array([embedding], dtype=np.float32)
    paper_indexIDMap.add_with_ids(embeds_np, ids_np)
    paper_library_df.loc[paper_library_df['faiss_id'] == paper_dict['faiss_id'], 'synopsis'] = synopsis


def index_section_synopsis(paper_dict, synopsis):
    global section_indexIDMap
    faiss_id = generate_faiss_id(lambda value: value in section_library_df.index)
    paper_title = paper_dict['title']
    paper_authors = paper_dict['authors'] 
    paper_abstract = paper_dict['summary']
    pdf_filepath = paper_dict['pdf_filepath']
    embedding = embedding_request(synopsis)
    ids_np = np.array([faiss_id], dtype=np.int64)
    embeds_np = np.array([embedding], dtype=np.float32)
    print(f'section synopsis length {len(synopsis)}')
    section_indexIDMap.add_with_ids(embeds_np, ids_np)
    synopsis_dict = {"faiss_id":faiss_id,
                     "paper_id": paper_dict["faiss_id"],
                     "synopsis": synopsis,
                     }
    section_library_df.loc[len(section_library_df)]=synopsis_dict
    return faiss_id

def index_paper(paper_dict):
    paper_title = paper_dict['title']
    paper_authors = paper_dict['authors'] 
    paper_abstract = paper_dict['summary']
    pdf_filepath = paper_dict['pdf_filepath']
    paper_faiss_id = generate_faiss_id(lambda value: value in paper_library_df.faiss_id)
    extract = create_chunks_grobid(pdf_filepath)
    if extract is None:
        return None
    print(f"grobid extract keys {extract.keys()}")
    text_chunks = [paper_abstract]+extract['sections']
    print(f"Summarizing each chunk of text {len(text_chunks)}")
    section_prompt = """Given this abstract of a paper:
    
<ABSTRACT>
{{$abstract}}
</ABSTRACT>

and this overall paper synopsis generated from the previous sections:

<PAPER_SYNOPSIS>
{{$paper_synopsis}}
</PAPER_SYNOPSIS>

Generate a synposis of this next section of text from the paper. The section synopsis should include the central argument of the section as it relates to the abstract and previous sections, together with all key points supporting that central argument. A key point might be any observation, statement, fact, inference, question, hypothesis, conclusion, or decision relevant to the central argument of the text.

<TEXT>
{{$text}}
</TEXT>

End your synopsis with:
</SYNOPSIS>
"""

    paper_prompt = """Your task is to create a detailed and comprehensive synopsis of the paper so far given the title, abstract, partial paper synopsis generated from previous sections, (if present) and the next section synopsis. The new version of the paper synopsis should be up to  1800 words in length. The goal is to represent the overall paper accurately and in-depth, capturing the overall argument along with essential statements, methods, observations, inferences, hypotheses, and conclusions. 
List of all key points in the paper, including all significant statements, methods, observations, inferences, hypotheses, and conclusions that support the core argument. 

<TITLE>
{{$title}}
<TITLE>

<ABSTRACT>
{{$abstract}}
</ABSTRACT>

<PARTIAL_PAPER_SYNOPSIS>
{{$paper_synopsis}}
</PARTIAL_PAPER_SYNOPSIS>

<SECTION SYNOPSIS>
{{$section}}
</SECTION SYNOPSIS>

Please ensure the synopsis provides depth while removing redundant or superflous detail, ensuring that no important aspect of the paper's argument, observations, methods, findings, or conclusions is included in the list of key points.
End your synopsis response as follows:

</UPDATED_SYNOPSIS>
"""
    section_synopses = []; section_ids = []
    paper_synopsis = ''
    with tqdm(total=len(text_chunks)) as pbar:
        for text_chunk in text_chunks:
            if len(text_chunk) < 16:
                continue
            print(f'index_paper extracting synopsis from chunk of length {len(text_chunk)}')
            prompt = [SystemMessage(section_prompt),
                      AssistantMessage('<SYNOPSIS>\n')
                      ]
            #response = llm.ask('', prompt, client=openAIClient, template=OPENAI_MODEL3, temp=0.05)
            response = llm.ask({"abstract":paper_abstract,
                                "paper_synopsis":paper_synopsis,
                                "text":text_chunk},
                               prompt,
                               template=OS_MODEL,
                               max_tokens=500,
                               temp=0.05)
            pbar.update(1)
            if response is not None:
                if '</SYNOPSIS>' in response:
                    response = response[:response.find('</SYNOPSIS>')]
                    print(f'index_paper result len {len(response)}')
                id = index_section_synopsis(paper_dict, response)
                section_synopses.append(response)
                section_ids.append(id)

                #
                ### now update paper synopsis with latest section synopsis
                ###   why not with entire section?
                #response = llm.ask('', messages, client = openAIClient, max_tokens=1000, template=OPENAI_MODEL3, temp=0.1)
                paper_messages = [SystemMessage(paper_prompt),
                                  AssistantMessage('<UPDATED_SYNOPSIS>')
                                  ]
                paper_response = llm.ask({"title":paper_title,
                                          "abstract":paper_abstract,
                                          "paper_synopsis":paper_synopsis,
                                          "section":section_synopses[-1]},
                                         paper_messages,
                                         template=OS_MODEL,
                                         max_tokens=1200,
                                         temp=0.1)
                end_idx = response.rfind('/UPDATED_SYNOPSIS')
                if end_idx < 0:
                    end_idx = len(response)
                    paper_synopsis = response[:end_idx-1]
    index_paper_synopsis(paper_dict, paper_synopsis)
    # forget this for now, we can always retrieve sections by matching on paper_id in the section_library
    #row = paper_library_df[paper_library_df['faiss_id'] == paper_dict['faiss_id']]
    #paper_library_df.loc[paper_library_df['faiss_id'] == paper_dict['faiss_id'], 'section_ids'] = section_ids
    save_synopsis_data()
    print(f'indexed {paper_title}, {len(section_synopses)} sections')
    return paper_synopsis, paper_faiss_id, section_synopses, section_ids


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding = embedding_request(text=str(query))
    strings_and_relatednesses = [
        (row, relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    print(strings[0])
    return strings[:top_n]

def create_chunks_grobid(pdf_filepath):
    #url = "http://localhost:8070/api/processFulltextDocument"
    url = "http://192.168.1.160:8070/api/processFulltextDocument"
    print(f'create_chunks_grobid pdf: {pdf_filepath}')
    pdf_file= {'input': open(pdf_filepath, 'rb')}
    extract = {"title":'', "authors":'', "abstract":'', "sections":[]}
    response = requests.post(url, files=pdf_file)
    if response.status_code == 200:
        with open('test.tei.xml', 'w') as t:
            t.write(response.text)
    else:
        print(f'grobid error {response.status_code}')
        return None
    xml_content = response.text
    # Parse the XML
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.fromstring(xml_content.encode('utf-8'))
    # Extract title
    title = tree.xpath('.//tei:titleStmt/tei:title[@type="main"]', namespaces=ns)
    title_text = title[0].text if title else 'Title not found'
    
    # Extract authors
    authors = tree.xpath('.//tei:teiHeader//tei:fileDesc//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:persName', namespaces=ns)
    authors_list = [' '.join([name.text for name in author.xpath('.//tei:forename | .//tei:surname', namespaces=ns)]) for author in authors]
    
    # Extract abstract
    abstract = tree.xpath('.//tei:profileDesc/tei:abstract//text()', namespaces=ns)
    abstract_text = ''.join(abstract).strip()
    
    # Extract major section titles
    # Note: Adjust the XPath based on the actual TEI structure of your document
    section_titles = tree.xpath('./tei:text/tei:body/tei:div/tei:head', namespaces=ns)
    titles_list = [title.text for title in section_titles]
    body_divs = tree.xpath('./tei:text/tei:body/tei:div', namespaces=ns)
    # Print extracted information
    extract["title"]=title_text
    print("Title:", title_text)
    extract["authors"]=', '.join(authors_list)
    print("Authors:", extract["authors"])
    extract["abstract"]=abstract_text
    print("Abstract:", len(abstract_text))
    print("Section titles:", len(titles_list))
    print("body divs:", len(titles_list))
    sections = []
    for element in body_divs:
        all_text = element.xpath('.//text()')
        # Combine text nodes into a single string
        combined_text = ''.join(all_text)
        print("Section text:", len(combined_text))
        if len(combined_text) > 7:
            sections.append(combined_text)
    extract["sections"] = sections
    return extract


def sbar_as_text(sbar):
    return f"\n{sbar['needs']}\nBackground:\n{sbar['background']}\nReview:\n{sbar['observations']}\n"

def search(query, sbar={"needs":'', "background":'', "observations":''}, web=False):
    
    """Query is a *list* of keywords or phrases
    This function does the following:
    - Reads in the paper_library.parquet file in including the embeddings
    - optionally searches and retrieves an additional set of articles
       - Scrapes the text out of any new files 
         - separates the text into sections
         - creates synopses at the section and document level
         - creates an embed vector for each synopsis
         - adds entries to the faiss IDMap and the section_library
    - Finds the closest n faiss IDs to the user's query
    - returns the section synopses and section_library and paper_library entries for the found items"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    #get_ articles does arxiv library search. This should be restructured
    results = []
    i = 0
    while search and len(results) == 0 and i < 2:
        results = get_articles(random.sample(query, min(8, max(1, len(query)-1))))
        i+=1
        print(f"arxiv search returned {len(results)} papers")
        for paper in results:
            print(f"    title: {paper['title']}")
    sys.exit(0)
    for paper in results:
        index_paper(paper)

    # arxiv search over, now search faiss
    paper_summaries = search_sections(query, sbar, top_k=24)
    print(f'found {len(paper_summaries)} sections')
    print("Summarizing into overall summary")
    messages=[SystemMessage("You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers"),
              UserMessage(f"""Write a detailed and comprehensive synopsis of these scientific papers with respect to this query description:

<QUERY>
{sbar_as_text(sbar)}
</QUERY>

The synopsis should be about 1200 words in length. The goal is to present the integrated consensus on the query, capturing the overall argument along with essential statements, methods, observations, inferences, hypotheses, and conclusions. 
Also, create a list of all key points in the papers, including all significant statements, methods, observations, inferences, hypotheses, and conclusions that support the core consensus described. 

Summaries:
{paper_summaries}

Please ensure the synopsis provides depth while removing redundant or superflous detail, ensuring that no critical aspect of the papers argument, observations, methods, findings, or conclusions is included in the list of key points.
End your synopsis response as follows:

<End Synopsis>
"""),
              AssistantMessage("Synopsis:\n")
              ]
    response = llm.ask('', messages, client = openAIClient, max_tokens=1800, template=OPENAI_MODEL4, temp=0.1)
    #response = llm.ask('', messages, template=OS_MODEL, max_tokens=1200, temp=0.1)
    end_idx = response.rfind('End Synopsis')
    if end_idx < 0:
        end_idx = len(response)
    return(response[:end_idx-1])

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
    #search(["Direct Policy Optimization"], web=False)
    plan_search(web=True)
    
