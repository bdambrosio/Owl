import os, sys, re
import json
import spacy
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
from OwlCoT import LLM, ListDialog, generate_faiss_id
import OwlCoT as cot
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize


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

def extract_entities(text):
    topic = ''
    kwd_messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
"""),
                  UserMessage("""Your current task is to extract all keywords and named-entities (which may appear as acronyms) from the following research excerpt.

Respond using the following format:
<NAMED_ENTITIES>
Entity1
Entity2
...
</NAMED_ENTITIES>

If distinction between keyword, acronym, or named_entity is unclear, it is acceptable to list a term or phrase under multiple categories.

<RESEARCH EXCERPT>
{{$text}}
</RESEARCH EXCERPT>
"""),
                  AssistantMessage("<NAMED_ENTITIES>\n")
              ]
    
    response = cot.llm.ask({"text":text}, kwd_messages,  max_tokens=300, temp=0.1, eos='</NAMED_ENTITIES>')
    # remove all more common things
    keywords = []; response_entities = []
    if response is not None:
        end = response.lower().rfind ('</NAMED_ENTITIES>'.lower())
        if end < 1: end = len(response)+1
        response = response[:end]
        response_entities = response.split('\n')
    for entity in response_entities: 
        if entity.startswith('<NAMED_E') or entity.startswith('</NAMED_E'):
            continue
        zipf = wf.zipf_frequency(entity, 'en', wordlist='large')
        if zipf < 2.85 and entity not in keywords:
            keywords.append(entity)
    for word in extract_acronyms(text):
        zipf = wf.zipf_frequency(word, 'en', wordlist='large')
        if zipf < 2.85 and word not in keywords:
            keywords.append(word)
    #print(f'\nRewrite Extract_entities: {keywords}\n')
    return keywords

def entities_to_str(item_list):
    return '\n'.join(item_list)


# Dependency Parsing and Triple Extraction (simplified example)
"""for token in doc:
    if token.dep_ in ('nsubj', 'dobj', 'pobj') and token.head.pos_ == 'VERB':
        subject = [w for w in token.head.lefts if w.dep_ == 'nsubj']
        if subject:
            subject = subject[0]
            predicate = token.head
            obj = token
            print(f"Subject: {subject}, Predicate: {predicate}, Object: {obj}")
"""
"""
if __name__=='__main__':
    nlp = spacy.load("en_core_sci_scibert")
    cot = cot.OwlInnerVoice(None)

    text = "Diabetic chronic cutaneous ulcers (DCU) are one of the serious complications of diabetes mellitus, occurring mainly in diabetic patients with peripheral neuropathy. Recent studies have indicated that microRNAs (miRNAs/miRs) and their target genes are essential regulators of cell physiology and pathology including biological processes that are involved in the regulation of diabetes and diabetes‐related microvascular complications. in vivo and in vitro models have revealed that the expression of some miRNAs can be regulated in the inflammatory response, cell proliferation, and wound remodelling of DCU. Nevertheless, the potential application of miRNAs to clinical use is still limited. Here, we provide a contemporary overview of the miRNAs as well as their associated target genes and pathways (including Wnt/β‐catenin, NF‐κB, TGF‐β/Smad, and PI3K/AKT/mTOR) related to DCU healing. We also summarize the current development of drugs for DCU treatment and discuss the therapeutic challenges of DCU treatment and its future research directions."

    doc = nlp(text)
    for ent in doc.ents:
        print(ent, end='; ')
    print('\n')
    print(extract_entities(text))
            
    text = "hsa-miR-1227-3p, hsa-miR-193a-3p"

    print(f'\n\nTEXT: {text}')
    doc = nlp(text)
    for ent in doc.ents:
        print(ent, end='; ')
    print('\n')
    entities = extract_entities(text)
    print(extract_entities(text))

    prompt=[SystemMessage('the user will provide a list of NERs and a Class. Respond with those provided NERs that are either subclasses or members of the provided class.'),
            UserMessage('NERs:\n{{$ners}}\n\nClass:\n{{$class}}'),
            AssistantMessage(f'Response:')
            ]
    response = cot.llm.ask({'ners':entities, 'class':'miRNA'}, prompt)
    print(response)
"""    
import rdflib

# Create a Graph
g = rdflib.Graph()

# Parse in an RDF file hosted locally or at a URL
#g.parse("/home/bruce/Downloads/ontology/geno.owl", format="application/rdf+xml")
g.parse("/home/bruce/Downloads/ontology/go.owl", format="application/rdf+xml")
print('rdf loaded')

query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX go-plus: <http://purl.obolibrary.org/obo/go/extensions/go.owl#>

SELECT ?entity
WHERE {
  ?entity rdfs:label "miRNA"@en .
}
"""
qres = g.query(query)

print(qres)
for row in qres:
    print(str(row.entity))

print('miRNA entity query done')
sys.exit(0)
# Define your SPARQL query
sparql_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geno: <http://purl.obolibrary.org/obo/geno.owl#>
PREFIX go-plus: <http://purl.obolibrary.org/obo/go/extensions/go.owl#>
SELECT ?entity
WHERE {
  { ?entity rdfs:label "miRNA" .}
  UNION
  { ?subclass rdfs:subClassOf ?entity .}
}
"""

# Execute the query
qres = g.query(sparql_query)
print(f'query finished {len(qres)} rows returned')
# Process the results
labels = []
for row in qres:
    # Define the SPARQL query to get the labels
    sparql_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?label
WHERE {
  <%s> rdfs:label ?label .
}
""" % row.entity

    # Execute the query
    qres = g.query(sparql_query)
    
    # Process the results
    for row in qres:
        #print(f"Label: {row.label}")
        if str(row.label) not in labels:
            labels.append(str(row.label))
print(labels)
