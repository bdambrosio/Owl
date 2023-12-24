import json
import sys
import os
import time
import re
import traceback
import readline as rl
from fastapi import FastAPI
from alphawave.MemoryFork import MemoryFork
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.ChoiceResponseValidator import ChoiceResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.alphawaveTypes import PromptCompletionOptions
import alphawave_pyexts.llmsearch.google_search_concurrent as gs
import alphawave_pyexts.utilityV2 as ut
import warnings
from PyPDF2 import PdfReader
from itertools import zip_longest
import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import wordfreq as wf
from bs4 import BeautifulSoup
#from unstructured.partition.html import partition_html
history = {}


#client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
model = 'gpt-3.5-turbo'
memory = VolatileMemory()
functions = None
tokenizer = GPT3Tokenizer()

app = FastAPI()


@app.get("/search/")
#{self.query}&model={GPT4}&max_chars={max_tokens*4}')
async def search(query: str, model:str = 'gpt-3.5-turbo', max_chars: int = 1200):
  global client, memory, functions, tokenizer
  response_text = ''
  storeInteraction = True
  try:
    query_phrase, keywords = ut.get_search_phrase_and_keywords(client, query, model, memory, functions, tokenizer)
    google_result= \
      gs.search_google(query, gs.QUICK_SEARCH, query_phrase, keywords, client, model, memory, functions, tokenizer, max_chars)

    print(f'google_result:\n{google_result}')
    source_urls = []
    if type(google_result) is list:
      text = ''
      for item in google_result:
        text += item['text'].strip()+'\n'
        source_urls.append(item['url'])
      
      print(f'text from google_search\n{text}\n')
      prompt = Prompt([
        SystemMessage(f'summarize the following text, removing duplications, with respect to {query}'),
        UserMessage('Text:\n{{$input}}'),
        AssistantMessage(' ')
        ])
      prompt_options = PromptCompletionOptions(completion_type='chat', model=model, max_tokens=int(max_chars/4))
      summary = ut.run_wave (client, {"input": text}, prompt, prompt_options,
                               memory, functions, tokenizer, max_repair_attempts=1,
                               validator=DefaultResponseValidator())
      if type(summary) is dict and 'status' in summary.keys() and summary['status'] == 'success':
        return {"result":summary['message']['content'], "source_urls":source_urls}
      else: return {'result': 'failure'}
  except Exception as e:
    traceback.print_exc()
    return {"result":str(e)}

def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    tokenizer = GPT3Tokenizer()
    reader = PdfReader(filepath)
    meta = reader.metadata
    number_of_pages = len(reader.pages)
    info = f"""
    {filepath}: 
      Author: {meta.author}
      Creator: {meta.creator}
      Producer: {meta.producer}
      Subject: {meta.subject}
      Title: {meta.title}
      Number of pages: {number_of_pages}
    """
    print(info)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        page_text = page.extract_text() + f"\nPage Number: {page_number}"
        pdf_text += page_text
        print(f"Page Number: {page_number}, chars: {len(page_text)}")
    return info, pdf_text

def convert_title_to_unix_filename(title):
    filename = title.replace(' ', '_')
    # Remove or replace special characters
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
    filename = filename[:64]
    return filename+'.pdf'

def wait_for_download(directory):
    """
    Wait for the download to complete in the specified directory.
    Returns the filename of the downloaded file.
    """
    while True:
        for fname in os.listdir(directory):
            if not fname.endswith('.crdownload'):
                return fname
        time.sleep(1)  # Wait a bit before checking again

@app.get("/retrieve/")
async def retrieve(title: str, url: str, doc_type='str', max_chars: int = 4000):
  global client, memory, functions, tokenizer
  response = ''
  try:
    #query_phrase, keywords = ut.get_search_phrase_and_keywords(client, title, model, memory, functions, tokenizer)
    #keyword_weights = gs.compute_keyword_weights(keywords)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      chrome_options = Options()
      chrome_options.page_load_strategy = 'eager'
      chrome_options.add_argument("--headless")
      download_dir = "/home/bruce/Downloads/pdfs/"
      prefs = {"download.default_directory": download_dir,
         "download.prompt_for_download": False,
         "download.directory_upgrade": True,
         "plugins.always_open_pdf_externally": True}
      chrome_options.add_experimental_option("prefs", prefs)

      result = ''
      with webdriver.Chrome(options=chrome_options) as dr:
        print(f'*****setting page load timeout {20} {url}')
        dr.set_page_load_timeout(20)
        dr.get(url)
        if url.endswith('pdf') or doc_type=='pdf':
          downloaded_file = wait_for_download(download_dir)
          print("Downloaded file:", downloaded_file)

        response = dr.page_source
        dr.close()

        if url.endswith('pdf') or doc_type=='pdf':
          print(f'\nchat processing pdf\n')
          filename = convert_title_to_unix_filename(title)
          print(f'reading pdf {download_dir+downloaded_file}')
          pdf_info, pdf_text = read_pdf(download_dir+downloaded_file)
          return {"result": pdf_text, "info": pdf_info, "filepath":download_dir+downloaded_file}

      print(f'\nchat processing non-pdf\n')
      soup = BeautifulSoup(response, "html.parser")
      all_text = soup.get_text()
      # Remove script and style tags
      all_text = all_text.split('<script>')[0]
      all_text = all_text.split('<style>')[0]
      # Remove multiple newline chars
      final_text = re.sub(r'(\n){2,}', '\n', all_text)
      print(f'retrieve returning {len(final_text)} chars')
      return {"result":final_text}
  except selenium.common.exceptions.TimeoutException as e:
    print(str(e))
    return {"result":'timeout'}
  except Exception as e:
    traceback.print_exc()
    return {"result":str(e)}
        
  return {"result":final_text}
  
