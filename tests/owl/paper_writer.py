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
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
from OwlCoT import LLM, ListDialog, generate_faiss_id
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from Planner import Planner
from transformers import AutoTokenizer, AutoModel
import webbrowser

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

import semanticScholar as s2
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
    
def entities (paper_title, paper_outline, paper_summaries):
    kwd_messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{json.dumps(paper_outline, indent=2)}

"""),
                  UserMessage(f"""Your current task is to extract all keywords, acronyms, and named-entities from the The following set of research extracts. An ACRONYM can be recognized by the regular expression r"\b[A-Za-z]+(?:-[A-Za-z\d]*)+\b", that is, any sequence of upper or lower case letters followed by one or more groups of upper or lower case letters or digits, separated by dashes.

Respond in JSON using the following format:
{{"keywords": ["keywd1", "keywd2", ...],
 "acronyms": ["ACRN1", "mi-RNA-2a", ...],
 "named_entities": ["Named Entity1", "NamedEntity2", ...]
}}
If distinction between keyword, acronym, or named_entity is unclear, it is acceptable to list a term or phrase under multiple categories.

<RESEARCH EXCERPTS>
{paper_summaries}
</RESEARCH EXCERPTS>
"""),
                  AssistantMessage("")
              ]
    response_json = s2.cot.llm.ask('', kwd_messages, client=s2.cot.llm.openAIClient, template='gpt-4-1106-preview', max_tokens=2000, temp=0.1, validator=JSONResponseValidator())
    # remove all more common things
    keywords = []
    if 'keywords' in response_json:
        for word in response_json['keywords']:
            zipf = wf.zipf_frequency(word, 'en', wordlist='large')
            if zipf < 2.5 and word not in keywords:
                keywords.append(word)
            else:
                print(f'excluding {word} {zipf}')
    if 'acronyms' in response_json:
        for word in response_json['acronyms']:
            zipf = wf.zipf_frequency(word, 'en', wordlist='large')
            if zipf < 2.5 and word not in keywords:
                keywords.append(word)
            else: print(f'excluding {word} {zipf}')
    if 'named_entities' in response_json:
        for word in response_json['named_entities']:
            zipf = wf.zipf_frequency(word, 'en', wordlist='large')
            if zipf < 2.8 and word not in keywords:
                keywords.append(word)
            else: print(f'excluding {word} {zipf}')
    #print(f' extract keywords out {keywords}')

    print(f'\nKeywords: {keywords}\n')
    return keywords


def write_paper(paper_outline, section_outline, length=5000, inst='', topic='', paper_title='', abstract='', depth=0, parent_section_title='', parent_section_partial='', heading_1=None, heading_1_draft = '', num_rewrites=0):
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
        for subsection in section_outline['sections']:
            if heading_1 is None:
                heading_1 = subsection['title']
                heading_1_draft = ''
            print(f"subsection title {subsection['title']}")
            section += '\n'+ "  "*depth+parent_section_title+'.'+write_paper(paper_outline=paper_outline,
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
                                                                             heading_1_draft=section,
                                                                             num_rewrites=num_rewrites)
            if depth==0: sys.exit(0)
        return section
    elif 'title' in section_outline:
        # actuall llm call to write this terminal section
        section = section_outline['title']
        query = heading_1+', '+parent_section_title+' '+subsection_topic
        paper_summaries = s2.search(query)
        paper_summaries = '\n'.join(['Title: '+s[0]+'\n'+s[1] for s in paper_summaries])
        subsection_token_length = max(320,length) # no less than a paragraph
        
        #
        ### Write initial content
        #
        
        print(f"\nWriting:{section_outline['title']} length {length}\n  within {parent_section_title}\n     within{heading_1}\n covering {subsection_topic}")
        messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{json.dumps(paper_outline, indent=2)}

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
 - Write a concise, detailed text using known fact and the above research excepts, of about {length} tokens in length.
 - This section should cover the specific topic: {parent_section_title} - {subsection_topic}
 - You may refer to, but not repeat, prior section content in any text you produce.
 - The section should be about {subsection_token_length} words long. The goal is to present an integrated view of the assigned topic, highlighting controversy where present, and capturing the overall state of knowledge along with essential statements, methods, observations, inferences, hypotheses, and conclusions. This must be done in light of the place of this section or subsection within the overall paper.

 - Please ensure the section provides depth while removing redundant or superflous detail, ensuring that every critical aspect of the source argument, observations, methods, findings, or conclusions is included in the list of key points.
End the section as follows:

</DRAFT>
"""),
              AssistantMessage("<DRAFT>\n")
              ]
        response = s2.cot.llm.ask('', messages, client=s2.cot.llm.osClient, max_tokens=3*subsection_token_length, temp=0.1, eos='</DRAFT>')
        #response = s2.cot.llm.ask('', messages, client=s2.cot.llm.openAIClient, template='gpt-4-1106-preview', max_tokens=3*subsection_token_length, temp=0.1, eos='</DRAFT>')
        #response = llm.ask('', messages, template=OS_MODEL, max_tokens=1200, temp=0.1)
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
        keywds = entities (paper_title, paper_outline, paper_summaries)

        rewrite_prompt=f"""Your current task is to rewrite the text titled: '{section_outline['title']}'. The current draft is:
<DRAFT>
{draft}
</DRAFT>

This part is located in the:
'{query}' 
section of the paper. 
The goal of the rewrite is to present an integrated view of the topic in light of the role this part plays in the overall paper, highlighting controversy where present, and capturing the overall state of knowledge along with essential statements, methods, observations, inferences, hypotheses, and conclusions.

<ENTITIES>
{json.dumps(keywds, indent=2)}
</ENTITIES>

The new draft should cover the specific topic: <SECTION_TOPIC> {subsection_topic} </SECTION_TOPIC>
Reason step by step to determine the role of this section within the paper, and use that information in writing the new draft. 
Generate increasingly entity-dense drafts by iterating these steps {num_rewrites} times:
Step 1. Identify 3-5 Missing Entities from the excerpts which are missing from the previously generated text. 
Step 2. Write a new, denser draft of the same length which covers every Entity and detail from the previous text plus the Missing Entities.

A Missing Entity is: 
 - In the ENTITIES (keywords, acronyms, or named_entities) provided above;
 - Relevant to the topic and role the content is to fill;
 - not in the previous text; 
 - present in the Excerpts

Guidelines: 
 - Coherence: re-write the previous text to improve flow and make space for additional entities;
 - The content should become highly dense and concise yet self-contained, e.g., easily understood without the source Excerpts;
 - Missing entities can appear anywhere in the new report;
 - Never drop entities from the previous content. If space cannot be made, add fewer new entities. 
 - use the same number of words for each rewrite;
 - Answer in JSON. The JSON should be a list (length {num_rewrites}) of dictionaries whose keys are "Missing_Entities" and "Denser_Draft"
 - ensure the rewrite provides depth and all information from the previous draft. 
 - End the DRAFT as follows:

</DRAFT>
"""
        messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{json.dumps(paper_outline, indent=2)}
"""),
                  UserMessage(rewrite_prompt),
                  AssistantMessage("<DRAFT>\n")
                  ]
        #response = s2.cot.llm.ask('', messages, client=s2.cot.llm.openAIClient, template='gpt-3.5-turbo-16k', max_tokens=3*length, temp=0.1, eos='</DRAFT>')
        response = s2.cot.llm.ask('', messages, client=s2.cot.llm.osClient, max_tokens=1.5*length, temp=0.1, eos='</DRAFT>')
        #response = llm.ask('', messages, template=OS_MODEL, max_tokens=1200, temp=0.1)
        end_idx = response.rfind('</DRAFT>')
        if end_idx < 0:
            end_idx = len(response)
        rewrite = response[:end_idx-1]
        print(f'\nRewrite:\n{rewrite}\n')

    return rewrite


if __name__ == '__main__':
    outline = {
        "task":
        """write a detailed research survey on circulating miRNA and DNA methylation patterns as biomarkers for early stage lung cancer detection.""",
        "title":"Circulating miRNA and DNA methylation patterns as biomarkers for early stage lung cancer detection.",
        "sections":[
            {"title":"Introduction", "length": 600,
             "sections":[
                 {"title":"Overview of Lung Cancer",
                  "sections":[
                      {"title":"Prevalence and impact"},
                      {"title":"Importance of early detection for improved prognosis"},
                  ]
                  },
                 {"title":"Biomarkers in Cancer Detection",
                  "sections":[
                      {"title":"Definition and role of biomarkers in cancer"},
                      {"title":"Traditional biomarkers used in lung cancer", "length":400, "rewrites":3},
                  ],
                  },
                 {"title":"Emerging Biomarkers: miRNA and DNA Methylation",
                  "sections":[
                      {"title":"Explanation of miRNA and DNA methylation"},
                      {"title":"Potential advantages over traditional biomarkers"},
                  ],
                  }
             ]
             },
            
            {"title":"Circulating miRNA as Biomarkers", "length":2000,
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
                      {"title":"Specific miRNAs frequently dysregulated in lung cancer", "length":300, "rewrites":2},
                  ],
                  },
                 {"title":"Circulating miRNAs",
                  "sections":[
                      {"title":"Explanation of circulating miRNAs in blood and other bodily fluids"},
                      {"title":"Advantages as non-invasive biomarkers"},
                  ],
                  },
                 {"title":"Studies on miRNAs in Early Lung Cancer Detection",
                  "sections":[
                      {"title":"Summary of key research findings", "length":300, "rewrites":2},
                      {"title":"Analysis of sensitivity, specificity, and overall effectiveness"},
                  ],
                  },
             ]
             },
            
            {"title":"DNA Methylation Patterns as Biomarkers", "length":500,
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
                 {"title":"Detection of Methylation Patterns",
                  "sections":[
                      {"title":"Techniques used to detect DNA methylation"},
                      {"title":"Challenges in using methylation patterns for early detection"},
                  ],
                  },
                 {"title":"Research on Methylation Patterns in Early Lung Cancer",
                  "sections":[
                      {"title":"Overview of significant studies"},
                      {"title":"Discussion on the feasibility and accuracy"},
                  ],
                  }
             ]
             },
            
            {"title":"Comparative Analysis", 
             "dscp":"miRNA and DNA Methylation Biomarkers",
             "sections":[
                 {"title":"Comparison in terms of reliability, cost, and accessibility"},
                 {"title":"miRNA vs. DNA Methylation Biomarkers - Potential for combining both markers for improved detection",
                  "length":1000, "rewrites":3},
             ],
             },
            
            {"title":"Challenges and Future directions", "length":1000, "rewrites":2,
             "sections":[
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
            
            {"title":"Conclusion",
             "sections":[
                 {"title":"Summary of Current Understanding",
                  "dscp":"Recap of the potential of circulating miRNA and DNA methylation patterns in early lung cancer detection",
                  },
                 {"title":"Future Outlook",
                  "dscp": "The future role of these biomarkers in clinical practice and Final thoughts on the impact of early detection on lung cancer outcomes"},
             ]
             }
        ]
    }
    
    paper = write_paper(outline, section_outline=outline)
    print(f'\n\n\n\n{paper}')
    #print(json.dumps(outline, indent=2))
    
