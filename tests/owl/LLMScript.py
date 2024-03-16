import os, sys
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
import semanticScholar3 as s2
import rewrite as rw
import numpy as np
import pickle
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class LLMScript:
    def __init__ (self, interpreter, cot):
        self.interpreter = interpreter
        self.cot = cot
        s2.cot = cot
        rw.cot = cot
        self.wm = interpreter.wm
        print(f'\ncreated script runner!')
        
    def extract(self, paper_pd = None, paper_id=None, uri=None, instruction=None, dest=None, max_tokens=200, redo=False):
        # works with a paper row, a in-memory paper_library_df faiss_id, or a uri (http:// or file:///)
        if paper_pd is None:
            if paper_id is None:
                if uri is None:
                    raise ValueError('need paper_id or uri!')
                paper_pd = s2.get_paper_pd(uri=uri)
            else:
                paper_pd = s2.get_paper_pd(paper_id=paper_id)
            if paper_pd is None:
                    raise ValueError(f"cant find paper {paper_id}{uri}")
        else:
            paper_id = paper_pd['faiss_id']

        if instruction is not None:
            raise ValueError('instruction arg not allowed in this version!')

        texts = s2.get_paper_sections(paper_id = paper_id)
        # Note below is now an array of length 'dimensions', without headers
        resource = rw.shorten(texts, instruction, sections=rw.default_sections, max_tokens=max_tokens)
        self.wm.assign(dest, resource)
        return dict(zip(rw.default_sections, resource))

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process1(self, arg1, instruction, dest, max_tokens):
        resolved_arg= self.interpreter.resolve_arg(arg1)
        prompt = [SystemMessage(instruction+"""\n
Limit your response to {{$length}} words. Do not include any discursive or explanatory text. 
End your response with </Response>"""),
                  UserMessage('\n<CONTENT>\n{{$input}}\n</CONTENT>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"input":resolved_arg, "length":int(max_tokens)},
                            prompt, max_tokens=int(1.2*max_tokens), eos='</Response>')
        self.wm.assign(dest, query)
        return query

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process2(self, arg1, arg2, instruction, dest, max_tokens):
        resolved_arg1 = self.interpreter.resolve_arg(arg1)
        resolved_arg2 = self.interpreter.resolve_arg(arg2)
        #resolved_inst = instruction.replace(arg1, 'the user Input provided below')
        #resolved_inst = resolved_inst.replace(arg2, 'the user SecondInput provided below')
        prompt = [SystemMessage("""{{$instruction}}
Do not include any discursive or explanatory text. Limit your response to {{$length}} words.
End your response with </Response>"""),
                  UserMessage('\n<Text1>\n{{$text1}}\n</Text1>\n\n<Text2>\n{{$text2}}\n</Text2>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"instruction":instruction, "text1":resolved_arg1, "text2":resolved_arg2,
                                  "length":int(max_tokens/1.33)}, prompt, max_tokens=int(1.2*max_tokens), eos='</Response>')
        self.wm.assign(dest, query)
        return query
    

    def create_paper_summary(self, paper_id):
        """ create a short summary of a paper for use by embedder """
        summary = self.extract(paper_id=paper_id,
                        dest='$summary',
                        max_tokens=800
                        )
        print(f'fetched paper_id {paper_id}\n{summary}')
        if summary and type(summary) is dict:
            summary_text = '\n'.join(f"{key}\n{value}" for key, value in summary.items())
            s2.set_paper_field(paper_id, 'summary', summary_text)
            print(f" summary wm contents:\n{summary_text}")
            #self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$summary", "result":'$trash'}])
            return paper_id
        else:
            raise Exception (f'No summary returned from self.extract\n{summary}')
            
    def create_paper_extract(self, paper_id):
        """ create a long technical extract of a paper for use in RAG """
        extract = self.extract(paper_id=paper_id,
                        #instruction='extract the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                        dest='$extract',
                        max_tokens=3000
                        )
        print(f'created extract for paper_id {paper_id} len {len(extract)}')
        if extract and type(extract) is dict:
            extract_text = '\n'.join(f"{key}\n{value}" for key, value in extract.items())
            s2.set_paper_field(paper_id, 'extract', extract_text)
            print(f" summary wm contents:\n{extract_text}")
            #self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$extract", "result":'$trash'}])
            return paper_id
        else:
            raise Exception ('No extract returned from self.extract')
            

    def create_paper_novelty(self, paper_id):
        """ identify what is novel in a paper - Needs to be updated for dict return!"""
        # start with extract
        paper_id = self.fetch(paper_id=paper_id,
                        dest='$paper1',
                        max_tokens=3000
                        )
        #print(f'fetched paper_id {paper_id}')
        if not paper_id:
            return

        # now identify what is novel to llm
        self.process1(arg1='$paper1',
                      instruction="""Please identify the main concepts, entities, and key facts present in the text below. 
For each item you identify, indicate whether it is something you are already familiar with or if it represents new information to you.
                      
Provide your response in the following format:
 - Concept/Entity/Fact 1: [Familiar/New]
 - Concept/Entity/Fact 2: [Familiar/New]
 ...
 """,
                      dest='$familiarity_assesment',
                      max_tokens=2000
                      )
        
        self.process2(arg1='$paper1',
                      arg2='$familiarity_assesment',
                      instruction="""
    Given Text1 below and your assessment of familiarity with the main concepts, entities, and key facts (Text2):
    Please identify the specific information, findings, or ideas in the paper summary that you consider to be new, different, or complementary to your existing knowledge. Focus on elements that expand or enrich your understanding of the topic.
    
    Provide your response in the following format:
    - Novel Information 1: [Description]
    - Novel Information 2: [Description]
    ...
""",
                      dest='$novel_information',
                      max_tokens=2000
                      )
        self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$novel_information", "result":'$trash'}])
        self.process2(arg1='$paper1',
                      arg2='$novel_information',
                      instruction="""
    Given the paper provided in Text1 below and your identified novel information, findings, or ideas provided in Text2 below:
    Please generate an overview of the paper in which you prioritize and highlight the novel elements you identified. Ensure that the overview maintains the overall context and coherence of the paper's content while emphasizing the new and complementary information. While highlighting the novel elements of Text1, also be sure to include all other problem, method, data, limitation, and conclusion information in the text.
    
    Provide your summary in a clear and structured format, using paragraphs or bullet points as appropriate.
""",
                      dest='$overview',
                      max_tokens=2000
                      )
        
        si.wm.get('$overview')
        self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$overview", "result":'$trash'}])
        return self.wm.get('$novel_information')['item']


    
    def pre_process_paper(self, uri):
        ### extract and summarize a paper, add extract and summary to df
        # first call fetch, which creates full extract
        paper_id = s2.index_url(uri=uri)
        if paper_id is not None:
            self.create_paper_extract(paper_id)
            self.create_paper_summary(paper_id)
            # lattice add paper - tbd
        s2.save_paper_df()
        self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$paper1Summary", "result":'$trash'}])
        

    def update_extracts(self):
        #sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in s2.paper_library_df.iterrows():
            paper_id = row['faiss_id']
            if row['extract'] is None or len(row['extract']) < 1000:
                print(f" {index} {row['faiss_id']}")
                self.create_paper_extract(paper_id)
            if index % 10 ==0:
                print(f'saving...')
                s2.save_paper_df()
            

    def update_summaries(self):
        #sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in s2.paper_library_df.iterrows():
            paper_id = row['faiss_id']
            #if row['summary'] is None or len(row['summary'])< 32:
            print(f" {index} {row['faiss_id']}")
            self.create_paper_summary(paper_id)
            if index % 10 ==0:
                print(f'saving...')
                s2.save_paper_df()
            
        s2.save_paper_df()

    def umap(self):
        # Load the embeddings (example using digits dataset)
        papers = si.s2.aper_library_df
        embeddings = np.array([si.s2.embedding_request(paper, 'search_document: ') for paper in papers['summary']])
        labels = np.array([i for i in range(len(papers))])
        
    
        print(type(embeddings), embeddings.shape, type(labels), labels.shape)
        print(labels[:3])
        # Create a UMAP model
        umap_model = UMAP(n_components=2, random_state=42)
        
        # Fit and transform the embeddings
        reduced_embeddings = umap_model.fit_transform(embeddings)
        
        # Plot the reduced embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=10)
        plt.colorbar(label='Labels')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Visualization of Reduced Embeddings')
        plt.show()
        
if __name__=='__main__':
    cot = OwlCoT.OwlInnerVoice()
    print('created cot')
    interp = Interpreter.Interpreter(cot)
    print('created interp')
    si = LLMScript(interp, s2, cot)
    print('created si')
    #si.create_paper_summary(paper_id=12218601)
    #si.create_paper_extract(paper_id=12218601)
    #paper = si.s2.get_paper_pd(paper_id=12218601)
    #paper = si.s2.get_paper_pd(paper_id=12218601)
    #si.create_paper_novelty(93853867)
    
    si.process1(arg1=paper['summary'],
                instruction="""Respond with a question for which Text1, provided below, contains an answer. 
Reason step by step:
 - 1. Analyze Text1 to identify information not available from known fact or logical reasoning
 - 2. Verify hen formulate a question for which that information is the answer.
Respond only with the question and the Text1 passage containing the answer.""",
                dest='$questions',
                max_tokens=100
                )
    
    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$questions", "result":'$trash'}])
    #browse_lattice(root_node, nodes)
    
