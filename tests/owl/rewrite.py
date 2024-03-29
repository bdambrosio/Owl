import json, os, sys
import wordfreq as wf
import re
import pickle
from promptrix.VolatileMemory import VolatileMemory
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator

cot=None
KEGG_SYMBOLS = {}
kegg_symbols_filepath = 'owl_data/kegg_symbols.pkl'
if not os.path.exists(kegg_symbols_filepath):
    print(f"no kegg_symbols.pkl found! ")
    sys.exit(-1)
else:
    try:
        with open(kegg_symbols_filepath, 'rb') as pf:
            KEGG_SYMBOLS=pickle.load(pf)
    except Exception as e:
        print(f'failure to load kegg_symbols, repair or recreate\n  {str(e)}')
        sys.exit(-1)
        print(f"loaded {kegg_symbols_filepath}")

NER_CACHE = {}
ner_cache_filepath = 'owl_data/paper_writer_ner_cache.json'
if not os.path.exists(ner_cache_filepath):
    with open(ner_cache_filepath, 'w') as pf:
        json.dump(NER_CACHE, pf)
        print(f"created 'paper_ner_cache'")
else:
    try:
        with open(ner_cache_filepath, 'r') as pf:
            NER_CACHE=json.load(pf)
    except Exception as e:
        print(f'failure to load entity cache, repair or delete\n  {str(e)}')
        sys.exit(-1)
        print(f"loaded {ner_cache_filepath}")


def hyde(query):
    # rewrite a query as an answer
    pass

def literal_missing_ners(ners, draft):
    # identifies items in the ners_list that DO appear in the summaries but do NOT appear in the current draft.
    missing_ners = ners.copy()
    draft_l = draft.lower()
    for ner in ners:
        if ner.lower() in draft_l :
            missing_ners.remove(ner)
    #print(f'Missing ners from draft: {len(missing_ners)}')
    return missing_ners

def literal_included_ners(ners, text):
    # identifies items in the ners_list that DO appear in the current draft.
    included_ners = []
    text_l = text.lower()
    for ner in ners:
        if ner.lower() in text_l:
            included_ners.append(ner)
    #print(f'Missing ners from draft: {missing_ners}')
    return included_ners

def section_ners(id, text, template):
    global NER_CACHE
    int_id = str(int(id)) # json 'dump' writes these ints as strings, so they won't match reloaded items unless we cast as strings
    if int_id not in NER_CACHE:
        excerpt_items = extract_ners(text, title='', outline='', template=template)
        NER_CACHE[int_id]=excerpt_items
        with open(ner_cache_filepath, 'w') as pf:
            json.dump(NER_CACHE, pf)
    return NER_CACHE[int_id]

def paper_ners(paper_title, paper_outline, paper_summaries, ids,template):
    global NER_CACHE
    items = set()
    total=len(ids); cached=0
    for id, excerpt in zip(ids, paper_summaries):
        # an 'excerpt' is [title, text]
        int_id = str(int(id)) # json 'dump' writes these ints as strings, so they won't match reloaded items unless we cast as strings
        if int_id in NER_CACHE:
            cached += 1
        else:
            excerpt_items = extract_ners(excerpt[0]+'\n'+excerpt[1], title=paper_title, outline=paper_outline, template=template)
            NER_CACHE[int_id]=list(set(excerpt_items))
        items.update(NER_CACHE[int_id])
    print(f'ners total {total}, in cache: {cached}')
    with open(ner_cache_filepath, 'w') as pf:
        json.dump(NER_CACHE, pf)
    #print(f"wrote {ner_cache_filepath}")
    return items

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

def find_kegg_symbols(text):
    global KEGG_SYMBOLS
    symbols_found = set()
    words = set(text.split(' ')) # going to be lots of duplicates of 'a', 'the', etc
    for candidate in words:
        candidate = candidate.strip()
        #print(f'testing {candidate}')
        if candidate in KEGG_SYMBOLS or candidate.lower() in KEGG_SYMBOLS:
            print(f' found kegg symbol {candidate}')
            symbols_found.add(candidate)
    return symbols_found

def extract_ners(text, title=None, paper_topic=None, outline=None, template=None):
    topic = ''
    if title is not None:
        topic += title+'\n'
    if paper_topic is not None:
        topic += paper_topic+'\n'
    if outline is not None and type(outline) is dict:
        topic += json.dumps(outline, indent=2)
        
    kwd_messages=[SystemMessage("""Your task is to extract all NERs (Named-entities, relations, or keywords, which may appear as acronyms) important to the topic {{$topic}} that appear in the following Text.

<Text>
{{$text}}
</Text>

Respond using the following format:
<NERs>
NER1
NER2
...
</NERs>

"""),
                  AssistantMessage("<NERs>\n")
              ]
    
    response = cot.llm.ask({"topic":topic, "text":text}, kwd_messages, template=template, max_tokens=300, temp=0.1, eos='</NERs>')
    # remove all more common things
    keywords = []; response_ners = []
    if response is not None:
        end = response.lower().rfind ('</NERs>'.lower())
        if end < 1: end = len(response)+1
        response = response[:end]
        response_ners = response.split('\n')
    candidates = response_ners+extract_acronyms(text)
    cutoff = 2.85
    while len(keywords) == 0 and len(candidates) > 0:
        for candidate in candidates:
            if candidate.startswith('<NAMED_E') or candidate.startswith('</NAMED_E'):
                continue
            zipf = wf.zipf_frequency(candidate, 'en', wordlist='large')
            if zipf < cutoff and candidate not in keywords:
                keywords.append(candidate)
        if len(keywords) <=1:
            cutoff += .1
    #print(f'\nRewrite Extract_ners: {keywords}\n')
    ### now check membership in KEGG
    keywords = set(keywords) # eliminate duplicates
    keywords.update(find_kegg_symbols(text))
    return list(keywords)

def ners_to_str(item_list):
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


def extract(title, text, draft, instruction, ners, tokens, template=None):
    prompt_prefix = SystemMessage("""You are to perform an analysis task. 
Analyze and extract information from:

{{$text}}.

In support of the following Downstream Task:

{{$instruction}}

Note that your task at this time is information extraction:
1. Do NOT attempt the Downstream Task.
2. Do NOT include any introductory, discursive, or explantory phrases or text.
3. Respond with a lists of any problem(s) addressed, methods, data, information, claims, hypotheses, and conclusions contained in the text.
4. Limit your response to {{$tokens}} words. End your response with </END>""")

    if draft is not None and len(draft) > 0:
        prompt_prefix = SystemMessage("""Your task is to extend an analysis / information extraction with new text.
Given the analysis result from text provided so far:

<PreviousAnalysis>
{{$draft}}
</PreviousAnalysis>

Extract information from this new text:

<NewText>
{{$text}}.
</NewText>

In support of the following Downstream Task:

{{$instruction}}

Note that your task at this time is analysis / information extraction:
1. Do NOT attempt the Downstream Task.
2. Do NOT include any introductory, discursive, or explantory phrases or text.
3. Review the PreviousAnalysis. Respond with any additional problems, methods, data, information, claims, hypotheses, and conclusions contained in the new text.
4. Limit your response to {{$tokens}} words. End your response with </END>""")
        
    prompt = [prompt_prefix, 
              AssistantMessage("""Extract:\n""")
              ]
    extract = cot.llm.ask({"draft": draft, "instruction":instruction, "ners":ners, "tokens":int(tokens), "text":text},
                          prompt, max_tokens=tokens, eos='</END>')
    return extract


def extract_w_focus(text, section_name, section_draft, extract_topic, tokens, template=None):
    if section_draft is not None and len(section_draft) > 0:
        prefix_text = """Your task is to extend an analysis / information extraction on
{{$extract_topic}}
using the new Text provided below.

The information extracted on this topic from previous text is:

<PreviousAnalysis>
{{$section_draft}}
</PreviousAnalysis>

Respond ONLY with the updated information extraction on 
{{$extract_topic}}

You should include as much relevant information from the Previous Analysis as possible, while integrating new information into a coherent exposition.

The new text is: """
    else:
        prefix_text = """Your task is to analyze and extract information on
{{$extract topic}}
from the following Text: """

    prompt_text=prefix_text+"""
<Text>
{{$text}}.
</Text>

Your task at this time is information extraction:
1. Do NOT include any introductory, discursive, or explantory phrases or text.
2. Focus on identifying and responding with the information on: {{$extract_topic}}
3. Limit your response to {{$tokens}} words. End your response with </EXTRACT>"""

    prompt = [SystemMessage(prompt_text),
              AssistantMessage("""<EXTRACT>\n""")
              ]
    extract = cot.llm.ask({"section_draft": section_draft, "extract_topic":extract_topic, "tokens":int(tokens*.67), "text":text},
                          prompt, max_tokens=tokens, eos='</EXTRACT>')
    return extract


def write(paper_title, paper_outline, section_title, paper_summaries, section_topic, section_token_length,
          parent_section_title, heading_1_title, heading_1_draft, template):
    #
    ### Write initial content
    #
        
    messages=[SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers.
You are writing a paper titled:
{paper_title}
The outline for the full paper is:
{format_outline(paper_outline)}

"""),
              UserMessage(f"""Your current task is to write the part titled: '{section_title}'

<RESEARCH EXCERPTS>
{paper_summaries}
</RESEARCH EXCERPTS>


The {heading_1_title} section content up to this point is:

<{heading_1_title}_SECTION_CONTENT>
{heading_1_draft}
</{heading_1_title}_SECTION_CONTENT>

Again, your current task is to write the next part, titled: '{section_title}'

1. First reason step by step to determine the role of the part you are generating within the overall paper, 
2. Then generate the appropriate text, subject to the following guidelines:

 - Output ONLY the text, do NOT output your reasoning.
 - Write a dense, detailed text using known fact and the above research excepts, of about {section_token_length} words in length.
 - This section should cover the specific topic: {parent_section_title}: {section_topic}
 - You may refer to, but should not repeat, prior subsection content in text you produce.
 - Present an integrated view of the assigned topic, noting controversy where present, and capturing the overall state of knowledge along with essential statements, methods, observations, inferences, hypotheses, and conclusions. This must be done in light of the place of this section or subsection within the overall paper.
 - Ensure the section provides depth while removing redundant or superflous detail, ensuring that every critical aspect of the source argument, observations, methods, findings, or conclusions is included.
End the section as follows:

</DRAFT>
"""),
              AssistantMessage("<DRAFT>\n")
              ]
    response = cot.llm.ask('', messages, template=template, max_tokens=int(section_token_length), temp=0.1, eos='</DRAFT>')
    if response is None:
        print(f'\n*** REWRITE.py write llm response None!\n')
        return ''
    end_idx = response.rfind('</DRAFT>')
    if end_idx < 0:
        end_idx = len(response)
    draft = response[:end_idx]
    return draft

def rewrite(paper_title, section_title, draft, paper_summaries, ners, section_topic, section_token_length, parent_section_title,
            heading_1, heading_1_draft, template):
    missing_ners = literal_missing_ners(ners, draft)
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

    MESelectMessage = f"""You are rewriting the draft for the section titled: '{section_title}', to increase the density of relevant information it contains. Your current task is to select the most important missing ners from the current draft. The following ners in the source material above have been identified as missing in the current draft:

<MISSING_NERS>
{{{{$missing_ners}}}}
</MISSING_NERS>

The current DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

Respond with up to six of the most important missing ners to add to the draft.
The order of the missing ners in the list above is NOT representative of their importance!
Many or all of the listed missing ners may be irrelevant to the role of this section.
Rate missing ner importance by:
 - relevance to the section title. 
 - relevance to the existing draft.
 - the role of this section within the overall outline. 
Do not respond with more than six ners. 
Respond as follows:

missing_ner 1
missing_ner 2
missing_ner 3
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

Ners (key phrases, acronyms, or names-ners) to add to the above draft include:

<MISSING_NERS>
{{{{$missing_ners}}}}
</MISSING_NERS>

The rewrite must cover the specific topic: {section_topic}

<INSTRUCTIONS>
Follow these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, denser draft of the same length that includes all ners and information and detail from the previous draft and adds content for the MISSING_NERS listed above.
 
Your response include ONLY the rewritten draft, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>

</INSTRUCTIONS>
"""

    messages=[sysMessage,
              UserMessage(MESelectMessage),
              AssistantMessage("<ME>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":paper_summaries, "missing_ners": ners_to_str(missing_ners)}, messages, template=template, max_tokens=200, temp=0.1, eos='</ME>')
    if response is None or len(response) == 0:
        return draft
    end_idx = response.rfind('</ME>')
    if end_idx < 0:
        end_idx = len(response)
    response = response[:end_idx-1]
    add_ners = response.split('\n')[:6]
    # pick just the actual ners out of responses
    add_ners = literal_included_ners(missing_ners, '\n'.join(add_ners))
    # downselect summaries to focus on selected missing content
    top_n_texts = select_top_n_texts(paper_summaries.split('\n'), add_ners, 24)

    research_excerpts = ''
    for text in top_n_texts:
        research_excerpts += text[0]+'\n'
        
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":research_excerpts, "missing_ners": add_ners}, messages, template=template, max_tokens=int(1.5*section_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    return rewrite

def add_pp_rewrite(paper_title, section_title, draft, paper_summaries, ners, section_topic, section_token_length, parent_section_title, heading_1, heading_1_draft, template):

    missing_ners = literal_missing_ners(ners, draft)

    # add a new paragraph on new ners
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

    MESelectMessage = f"""You are rewriting the draft for the section titled: '{section_title}', to increase the density of relevant information it contains. Your current task is to select the most important missing ners from the current draft. The following ners in the source material above have been identified as missing in the current draft:

<MISSING_NERS>
{{{{$missing_ners}}}}
</MISSING_NERS>

The current DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

Respond with up to six of the most important missing ners to add to the draft.
The order of the missing ners in the list above is NOT representative of their importance!
Many or all of the listed missing ners may be irrelevant to the role of this section.
Rate missing ner importance by:
 - relevance to the section title. 
 - relevance to the existing draft.
 - the role of this section within the overall outline. 
Do not respond with more than six ners. 
Respond as follows:

missing_ner 1
missing_ner 2
missing_ner 3
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

Ners (key phrases, acronyms, or names-ners) to add to the above draft include:

<MISSING_NERS>
{{{{$missing_ners}}}}
</MISSING_NERS>

The new paragraph must be pertinent to the specific topic: {section_topic}, and fit as an extension of the current draft.

<INSTRUCTIONS>
Follow these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, dense, concise paragraph that adds content for the MISSING_NERS listed above.
 
Your response include ONLY the new paragraph, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>

</INSTRUCTIONS>
"""

    messages=[sysMessage,
              UserMessage(MESelectMessage),
              AssistantMessage("<ME>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":paper_summaries, "missing_ners": ners_to_str(missing_ners)}, messages, template=template, max_tokens=200, temp=0.1, eos='</ME>')
    if response is None or len(response) == 0:
        return draft
    end_idx = response.rfind('</ME>')
    if end_idx < 0:
        end_idx = len(response)
    response = response[:end_idx-1]
    add_ners = response.split('\n')[:6]
    add_ners = literal_included_ners(missing_ners, '\n'.join(add_ners))
    # downselect summaries to focus on selected missing content
    top_n_texts = select_top_n_texts(paper_summaries.split('\n'), add_ners, 24)

    research_excerpts = ''
    for text in top_n_texts:
        research_excerpts += text[0]+'\n'
        
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":research_excerpts, "missing_ners": add_ners}, messages, template=template, max_tokens=int(1.5*section_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    return draft + '\n'+rewrite

def depth_rewrite(paper_title, section_title, draft, paper_summaries, ners, section_topic, section_token_length, parent_section_title, heading_1, heading_1_draft, template):

    missing_ners = literal_missing_ners(ners, draft)
    sysMessage = SystemMessage(f"""You are a brilliant research analyst, able to see and extract connections and insights across a range of details in multiple seemingly independent papers. You write in a professional, objective tone
You are writing a paper titled: '{paper_title}'

The following is a set of published research extracts that may be useful in writing the draft of this part of the paper:

<RESEARCH EXCERPTS>
{{{{$paper_summaries}}}}
</RESEARCH EXCERPTS>

"""
)

    MESelectMessage = f"""You are rewriting the draft for the section titled: '{section_title}', to increase the density of relevant information it contains. Your current task is to select the most important ners in the current draft. The following ners have been identified as present in the current draft:

<NERS>
{{{{$ners}}}}
</NERS>

The current DRAFT is:

<DRAFT>
{{{{$draft}}}}
</DRAFT>

Respond with up to four of the most important ners in the draft.
The order of the ners in the list above is NOT representative of their importance!
Many or all of the listed ners may be irrelevant to the role of this draft.
Rate ner importance by:
 - relevance to the section title. 
 - relevance to the existing draft.
 - important information about this ner in the research excerpts omitted in the current draft.
 - the role of this section within the overall outline. 
Do not respond with more than four ners. 
Respond as follows:

ner 1
ner 2
ner 3
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

Ners (key phrases, acronyms, or names-ners) to add to the above draft include:

<NERS>
{{{{$ners}}}}
</NERS>

The rewrite must cover the specific topic: {section_topic}

<INSTRUCTIONS>
Follow these steps:
Step 1. Reason step by step to determine the role of this section within the paper.
Step 2. Write a new, denser draft of the same length that includes all ners and information and detail from the previous draft and adds depth for the NERS listed above.
 
Your response include ONLY the rewritten draft, without any explanation or commentary.

end the rewrite as follows:
</REWRITE>

</INSTRUCTIONS>
"""

    messages=[sysMessage,
              UserMessage(MESelectMessage),
              AssistantMessage("<AE>\n")
              ]
    #ners from research summaries mentioned in current draft
    included_ners = literal_included_ners(ners, draft)
    #select ners to expand
    print(f'\ntemplate {template}')
    response = cot.llm.ask({"draft":draft, "paper_summaries":paper_summaries, "ners": ners_to_str(included_ners)},
                           messages,
                           template=template,
                           max_tokens=200, temp=0.1, eos='</AE>')
    if response is None or len(response) == 0:
        return draft
    end_idx = response.rfind('</AE>')
    if end_idx < 0:
        end_idx = len(response)
    response = response[:end_idx-1]
    add_ners = response.split('\n')[:4]
                                             # llm response can include other junk, prune                                             
    add_ners = literal_included_ners(ners, '\n'.join(add_ners))

    # downselect summaries to focus on selected ners 
    top_n_texts = select_top_n_texts(paper_summaries.split('\n'), add_ners, 24)

    research_excerpts = ''
    for text in top_n_texts:
        research_excerpts += text[0]+'\n'
        

    # expand content on selected ners
    messages=[sysMessage,
              UserMessage(rewrite_prompt),
              AssistantMessage("<REWRITE>\n")
              ]
    response = cot.llm.ask({"draft":draft, "paper_summaries":research_excerpts, "ners": add_ners}, messages, template=template, max_tokens=int(1.5*section_token_length), temp=0.1, eos='</REWRITE>')
    if response is None or len(response) == 0:
        return draft
    rewrite = response
    end_idx = rewrite.rfind('</REWRITE>')
    if end_idx < 0:
        end_idx = len(rewrite)
        rewrite = rewrite[:end_idx-1]
    return rewrite


# the first set of extract sections defined, for research papers

default_sections = {"SubjectArea":{"instruction":'the primary scientific subject area of this text, both in broad scientific terms and the relevant subfield within which this work takes place?'},
                      "Problem":{"instruction":'the primary problem, task, or objective addressed.'},
                      "Approach":{"instruction":'the specific methods, techniques, algorithms, theoretical foundations, or design principles employed to address the problem or achieve the objectives.'},
                      "Analysis":{"instruction":'the assessment or evaluation reported in the work, including data analysis, experimental results, theoretical analysis, performance metrics, or comparative studies.'},
                      "Findings":{"instruction":'the main outcomes, insights, contributions, or findings reported in the work, such as new knowledge, algorithms, theoretical advancements, or performance improvements.'},
                      "Limitations":{"instruction":'the assumptions, constraints, limitations, or potential areas for further research and improvement identified in the work.'},
                      "Impact":{"instruction":'the practical applications, real-world impact, potential use cases, or implications of the work in relevant domains or industries.'},
                      "Context":{"instruction":'the relationship of the work to existing literature, the broader context of the field, and the novelty or significance of the contribution within that context.'}
                      }

def shorten(resources, focus='', sections=default_sections, max_tokens=80*len(default_sections)):
    """ rewrite a collection of resources (text strings) to a max length, given a focus """
    """ assumes resources in sequence? """
    num_sections = len(sections.keys())
    resources = [resource.strip() for resource in resources]
    resource = '\n'.join(resources)
    input_length = len(resource)
    context_size = cot.llm.context_size
    print(f'rw.shorten total in: {input_length} chars, limit: {max_tokens} tokens')
    if input_length < max_tokens*3: # context is in tokens, len is chars
        print(f'rw.shorten returning full text: {len(resource)} chars')
        return resource
    text_to_shorten = ''
    rewrite = ''
    target = 0 # max_tokens
    section_extracts = ['' for i in range(num_sections)]
    target = int(max_tokens/num_sections) # we rewrite each pass, so target is fixed size!

    for n, text in enumerate(resources):
        print(f'rw.shorten section in: {len(text)}')
        # process as much text as possible - assumes 'max_tokens' is about 1/3 context size
        if len(text) + len(text_to_shorten) < int(context_size*2): # note this is comparing chars to tokens, implicit divide by
            text_to_shorten += text+'\n'
            if n < len(resources)-1:
                # don't process yet if this isn't the last section of text
                continue
        
        for e, key in enumerate(sections.keys()):
            print(f'{e}, {key} {type(sections[key])}')
            section_extracts[e] = extract_w_focus(text_to_shorten, key, section_extracts[e],
                                                    sections[key]['instruction'],
                                                    target)
        #print(f'\n{key}:\n{extracts[key]}\n')

        text_to_shorten = text # start with this next time.
    print(f'rw.shorten out: {len(str(section_extracts))} chars')
    return section_extracts
