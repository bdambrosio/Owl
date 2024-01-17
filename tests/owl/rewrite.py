import json
import tiktoken
from tqdm import tqdm
from termcolor import colored
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
import re
from transformers import AutoTokenizer, AutoModel
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

# cot will be set by invoker
cot=None

def hyde(query):
    # rewrite a query as an answer
    pass

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
            excerpt_items = extract_entities(excerpt[0]+'\n'+excerpt[1], title=paper_title, outline=paper_outline, template=template)
            entity_cache[int_id]=excerpt_items
            items.extend(entity_cache[int_id])
    print(f'entities total {total}, in cache: {cached}')
    with open(entity_cache_filepath, 'w') as pf:
        json.dump(entity_cache, pf)
    #print(f"wrote {entity_cache_filepath}")
    return list(set(items))

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

def extract_entities(text, title=None, paper_topic=None, outline=None, template=None):
    topic = ''
    if title is not None:
        topic += title+'\n'
    if paper_topic is not None:
        topic += paper_topic+'\n'
    if outline is not None and type(outline) is dict:
        topic += json.dumps(outline, indent=2)
        
    kwd_messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
"""),
                  UserMessage("""Your current task is to extract all keywords and named-entities (which may appear as acronyms) important to the topic {{$topic}} from the following research excerpt.

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
    
    response = cot.llm.ask({"topic":topic, "text":text}, kwd_messages, template=template, max_tokens=300, temp=0.1, eos='</NAMED_ENTITIES>')
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


def rewrite(paper_title, section_title, draft, paper_summaries, entities, section_topic, section_token_length, parent_section_title, heading_1, heading_1_draft, template):
    missing_entities = literal_missing_entities(entities, draft)
    sysMessage = SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers. You write in a professional, objective tone.
You are writing a paper titled:
{paper_title}

The following is a set of published research extracts that may be useful in writing the draft of this part of the paper:

<RESEARCH EXCERPTS>
{{{{$paper_summaries}}}}
</RESEARCH EXCERPTS>

The {heading_1} section content up to this point is:

<PRIOR_SECTION_CONTENT>
{heading_1_draft}
</PRIOR_SECTION_CONTENT>

"""
)

    MESelectMessage = f"""You are rewriting the draft for the section titled: '{section_title}', to increase the density of relevant information it contains. Your current task is to select the most important missing entities from the current draft. The following entities in the source material above have been identified as missing in the current draft:

<MISSING_ENTITIES>
{{{{$missing_entities}}}}
</MISSING_ENTITIES>

The current DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

Respond with up to six of the most important missing entities to add to the draft.
The order of the missing entities in the list above is NOT representative of their importance!
Many or all of the listed missing entities may be irrelevant to the role of this section.
Rate missing entity importance by:
 - relevance to the section title. 
 - relevance to the existing draft.
 - the role of this section within the overall outline. 
Do not respond with more than six entities. 
Respond as follows:

missing_entity 1
missing_entity 2
missing_entity 3
...

Respond only with the above list. Do not include any commentary or explanatory material.
End your response with:

</ME>
"""

    rewrite_prompt=f"""Your current task is to rewrite the DRAFT for the part titled: '{section_title}', to increase the density of relevant information it contains. You write in a professional, objective tone.
The current version of the DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

This DRAFT is for the '{section_title}' section.

Entities (key phrases, acronyms, or names-entities) to add to the above draft include:

<MISSING_ENTITIES>
{{{{$missing_entities}}}}
</MISSING_ENTITIES>

The rewrite must cover the specific topic: {section_topic}

<INSTRUCTIONS>
Follow these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, denser draft of the same length that includes all entities and information and detail from the previous draft and adds content for the MISSING_ENTITIES listed above.
 
Your response include ONLY the rewritten draft, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>

</INSTRUCTIONS>
"""

    messages=[sysMessage,
              UserMessage(MESelectMessage),
              AssistantMessage("<ME>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":paper_summaries, "missing_entities": entities_to_str(missing_entities)}, messages, template=template, max_tokens=200, temp=0.1, eos='</ME>')
    if response is None or len(response) == 0:
        return draft
    end_idx = response.rfind('</ME>')
    if end_idx < 0:
        end_idx = len(response)
    response = response[:end_idx-1]
    add_entities = response.split('\n')[:6]
    # pick just the actual entities out of responses
    add_entities = literal_included_entities(missing_entities, '\n'.join(add_entities))
    # downselect summaries to focus on selected missing content
    top_n_texts = select_top_n_texts(paper_summaries.split('\n'), add_entities, 24)

    research_excerpts = ''
    for text in top_n_texts:
        research_excerpts += text[0]+'\n'
        
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":research_excerpts, "missing_entities": add_entities}, messages, template=template, max_tokens=int(1.5*section_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    return rewrite

def add_pp_rewrite(paper_title, section_title, draft, paper_summaries, entities, section_topic, section_token_length, parent_section_title, heading_1, heading_1_draft, template):

    missing_entities = literal_missing_entities(entities, draft)

    # add a new paragraph on new entities
    sysMessage = SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers. You write in a professional, objective tone.
You are writing a paper titled:
{paper_title}

The following is a set of published research extracts that may be useful in writing the draft of this part of the paper:

<RESEARCH_EXCERPTS>
{{{{$paper_summaries}}}}
</RESEARCH_EXCERPTS>

The {heading_1} section content up to this point is:

<PRIOR_SECTION_CONTENT>
{heading_1_draft}
</PRIOR_SECTION_CONTENT>

"""
)

    MESelectMessage = f"""You are rewriting the draft for the section titled: '{section_title}', to increase the density of relevant information it contains. Your current task is to select the most important missing entities from the current draft. The following entities in the source material above have been identified as missing in the current draft:

<MISSING_ENTITIES>
{{{{$missing_entities}}}}
</MISSING_ENTITIES>

The current DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

Respond with up to six of the most important missing entities to add to the draft.
The order of the missing entities in the list above is NOT representative of their importance!
Many or all of the listed missing entities may be irrelevant to the role of this section.
Rate missing entity importance by:
 - relevance to the section title. 
 - relevance to the existing draft.
 - the role of this section within the overall outline. 
Do not respond with more than six entities. 
Respond as follows:

missing_entity 1
missing_entity 2
missing_entity 3
...

Respond only with the above list. Do not include any commentary or explanatory material.
End your response with:

</ME>
"""

    rewrite_prompt=f"""Your current task is to expand the DRAFT for the part titled: '{section_title}', to increase the density of relevant information it contains. 
The current version of the DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

This DRAFT is for the: '{section_title}' section within:\n {parent_section_title} 

Entities (key phrases, acronyms, or names-entities) to add to the above draft include:

<MISSING_ENTITIES>
{{{{$missing_entities}}}}
</MISSING_ENTITIES>

The new paragraph must be pertinent to the specific topic: {section_topic}, and fit as an extension of the current draft.

<INSTRUCTIONS>
Follow these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, dense, concise paragraph that adds content for the MISSING_ENTITIES listed above.
 
Your response include ONLY the new paragraph, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>

</INSTRUCTIONS>
"""

    messages=[sysMessage,
              UserMessage(MESelectMessage),
              AssistantMessage("<ME>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":paper_summaries, "missing_entities": entities_to_str(missing_entities)}, messages, template=template, max_tokens=200, temp=0.1, eos='</ME>')
    if response is None or len(response) == 0:
        return draft
    end_idx = response.rfind('</ME>')
    if end_idx < 0:
        end_idx = len(response)
    response = response[:end_idx-1]
    add_entities = response.split('\n')[:6]
    add_entities = literal_included_entities(missing_entities, '\n'.join(add_entities))
    # downselect summaries to focus on selected missing content
    top_n_texts = select_top_n_texts(paper_summaries.split('\n'), add_entities, 24)

    research_excerpts = ''
    for text in top_n_texts:
        research_excerpts += text[0]+'\n'
        
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":research_excerpts, "missing_entities": add_entities}, messages, template=template, max_tokens=int(1.5*section_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    return draft + '\n'+rewrite

def depth_rewrite(paper_title, section_title, draft, paper_summaries, entities, section_topic, section_token_length, parent_section_title, heading_1, heading_1_draft, template):

    missing_entities = literal_missing_entities(entities, draft)
    sysMessage = SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers. You write in a professional, objective tone
You are writing a paper titled: '{paper_title}'

The following is a set of published research extracts that may be useful in writing the draft of this part of the paper:

<RESEARCH EXCERPTS>
{{{{$paper_summaries}}}}
</RESEARCH EXCERPTS>

"""
)

    MESelectMessage = f"""You are rewriting the draft for the section titled: '{section_title}', to increase the density of relevant information it contains. Your current task is to select the most important entities in the current draft. The following entities have been identified as present in the current draft:

<ENTITIES>
{{{{$entities}}}}
</ENTITIES>

The current DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

Respond with up to four of the most important entities in the draft.
The order of the entities in the list above is NOT representative of their importance!
Many or all of the listed entities may be irrelevant to the role of this draft.
Rate entity importance by:
 - relevance to the section title. 
 - relevance to the existing draft.
 - important information about this entity in the research excerpts omitted in the current draft.
 - the role of this section within the overall outline. 
Do not respond with more than four entities. 
Respond as follows:

entity 1
entity 2
entity 3
...

Respond only with the above list. Do not include any commentary or explanatory material.
End your response with:

</ME>
"""

    rewrite_prompt=f"""Your current task is to rewrite the DRAFT for the part titled: '{section_title}', to increase the density of relevant information it contains. You write in a professional, objective tone.
The current version of the DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

This DRAFT is for the '{section_title} section within:\n {parent_section_title} 

Entities (key phrases, acronyms, or names-entities) to add to the above draft include:

<ENTITIES>
{{{{$entities}}}}
</ENTITIES>

The rewrite must cover the specific topic: {section_topic}

<INSTRUCTIONS>
Follow these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, denser draft of the same length that includes all entities and information and detail from the previous draft and adds depth for the ENTITIES listed above.
 
Your response include ONLY the rewritten draft, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>

</INSTRUCTIONS>
"""

    messages=[sysMessage,
              UserMessage(MESelectMessage),
              AssistantMessage("<AE>\n")
              ]
    #entities from research summaries mentioned in current draft
    included_entities = literal_included_entities(entities, draft)
    #select entities to expand
    print(f'\ntemplate {template}')
    response = cot.llm.ask({"draft":draft, "paper_summaries":paper_summaries, "entities": entities_to_str(included_entities)},
                           messages,
                           template=template,
                           max_tokens=200, temp=0.1, eos='</AE>')
    if response is None or len(response) == 0:
        return draft
    end_idx = response.rfind('</AE>')
    if end_idx < 0:
        end_idx = len(response)
    response = response[:end_idx-1]
    add_entities = response.split('\n')[:4]
                                             # llm response can include other junk, prune                                             
    add_entities = literal_included_entities(entities, '\n'.join(add_entities))

    # downselect summaries to focus on selected entities 
    top_n_texts = select_top_n_texts(paper_summaries.split('\n'), add_entities, 24)

    research_excerpts = ''
    for text in top_n_texts:
        research_excerpts += text[0]+'\n'
        

    # expand content on selected entities
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":research_excerpts, "entities": add_entities}, messages, template=template, max_tokens=int(1.5*section_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    return rewrite

