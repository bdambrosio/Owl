import os, sys
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
import semanticScholar3 as s2
import rewrite as rw
import OwlCoT
import Interpreter
import numpy as np
import pickle
"""
Here's a proposed approach to build a lattice of abstractions from a collection of research papers and use it to process and integrate new papers:

Process each paper in the collection using your existing digestion process to extract the central topic, methods, data, inferences, and conclusions.
Use the LLM to generate a high-level summary of each paper based on the extracted information. This summary should capture the key contributions and place the paper in the context of the broader field.
Cluster the paper summaries based on their semantic similarity. You can use techniques like latent semantic analysis, topic modeling (e.g., LDA), or embedding-based clustering (e.g., using BERT embeddings). This will group papers that share similar themes, methods, or findings.
For each cluster, use the LLM to generate a coherent synthesis that captures the common threads and key ideas. This synthesis should abstract away from the details of individual papers and provide a higher-level perspective on the subfield or research theme represented by the cluster.
Recursively cluster the cluster syntheses to create higher levels of abstraction. At each level, generate a new synthesis that integrates the key ideas from the level below. Continue this process until you reach a single, top-level synthesis that captures the overarching themes and state of knowledge in the entire field.
At each level of the lattice, from individual papers up to the top-level synthesis, use the LLM to generate a structured representation of the key entities, relationships, and propositions. This could take the form of a knowledge graph or a set of predicate-argument structures. This structured representation will allow for more precise reasoning and integration of new information.
When a new paper arrives, process it using the same digestion process to extract its key components. Then, use a combination of semantic similarity and the LLM's reasoning capabilities to identify where in the lattice the new paper fits best. This could involve finding the most similar cluster at each level and generating a rationale for why the paper belongs there.
Once the best fit for the new paper is identified, use the LLM to generate an updated synthesis that incorporates the paper's contributions. This may involve modifying the existing synthesis, adding new entities and relationships to the structured representation, or even splitting a cluster if the new paper introduces a sufficiently distinct perspective.
Propagate the changes upward through the lattice, updating the syntheses and structured representations at each level to reflect the new information. The LLM can be used to ensure that the updates maintain coherence and consistency with the existing knowledge.
Finally, use the LLM to generate a human-readable report summarizing how the new paper fits into and extends the current understanding of the field, based on its place in the updated lattice.
The key advantages of this approach are:

It leverages the strengths of both the LLM (language understanding, generation, reasoning) and structured representations (precise knowledge capture, systematic reasoning).
It provides a multi-resolution view of the field, allowing users to zoom in on specific subfields or zoom out to get a big-picture perspective.
It supports incremental updating and refinement of the knowledge representation as new papers arrive, ensuring that it stays current with the latest research.
It generates human-readable syntheses and reports, making the knowledge accessible and usable for a wide range of stakeholders.
"""


import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

si = None

class ClusterNode:
    # Written by Claude3 to my spec
    """... following a fairly extensive dialog about building abstractions over paper collections:
    this code should return a lattice of the built clusters, with, at each node in the lattice: 
    the node id, it's parent cluster, it's child clusters, it's summary, and it's embedding. 
    For leaf nodes the node should also contain the paper_ids clustered at that node. 
    Input parameters should be:
      the set of papers 
         (assume they are pandas rows with paper_id, summary, and embedding_vector fields), 
      the number of levels, 
      and, for each level, the cluster-size requiring splitting into subclusters at the next level. 
    If I have provided enough information, please write the code, otherwise ask clarifying questions.
"""
    def __init__(self, node_id, parent_id, level, summary, embedding, paper_ids=None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.level = level
        self.summary = summary
        self.embedding = embedding
        self.paper_ids = paper_ids if paper_ids else []
        self.children = []

def get_sibling_nodes(node, cluster_nodes):
    if node.parent_id is None:
        return []  # Root node has no siblings
    
    parent_node = cluster_nodes[node.level - 1][node.parent_id]
    sibling_ids = [child_id for child_id in parent_node.children if child_id != node.node_id]
    sibling_nodes = [cluster_nodes[node.level][sibling_id] for sibling_id in sibling_ids]
    
    return sibling_nodes


def generate_node_summary(node, cluster_nodes, papers, max_children=20):
    global si
    if node.paper_ids:
        print(f'generate node summary for {node.node_id} of {node.parent_id} including {node.paper_ids}')
        for paper_id in node.paper_ids:
            print(f"  {papers.loc[papers['faiss_id'] == paper_id, 'summary'].iloc[0][:64]}")
        item_len = int(16000/len(node.paper_ids))
        summaries = [papers.loc[papers['faiss_id'] == paper_id, 'summary'].iloc[0][:item_len] for paper_id in node.paper_ids]
    else:
        # If the node doesn't have paper IDs, it's an internal node
        child_summaries = []
        for child_id in node.children:
            child_node = cluster_nodes[node.level + 1][child_id]
            child_summaries.append(child_node.summary)

        # Limit the number of child summaries to max_children
        summaries = child_summaries[:max_children]

    # Concatenate the summaries and truncate if necessary
    concatenated_summaries = ' '.join(summaries)
    max_tokens = 8000  # Adjust the maximum token limit as needed
    if len(concatenated_summaries.split()) > max_tokens*2:
        truncated_summaries = ' '.join(concatenated_summaries.split()[:max_tokens])
        concatenated_summaries = truncated_summaries + '...'

    si.process1(arg1=concatenated_summaries,
                  instruction='The content provided below is a collection of summaries of papers or subtopics. Provide a synopsis of this content, focusing on the overall subfield represented by this collection. Include an overview of the topics or problem addressed, methods used, data presented, inferences or claims made, and conclusions drawn. Remove redundant information where possible. Your response should be a coherent narrative representing the collection as a whole and the important contributions in this collection.',
                  dest='$paperSummary',
                  max_tokens=800
                  )
    return si.wm.get('$paperSummary')['item']


def build_cluster_lattice(papers, num_levels, distance_thresholds):
    # Extract embeddings from papers
    embeddings = np.array([si.s2.embedding_request(paper, 'search_document: ') for paper in papers['summary']])

    def split_cluster(parent_node, parent_embeddings, parent_papers, level):
        if level >= num_levels:
            return

        # Perform hierarchical clustering for the current cluster
        Z = linkage(parent_embeddings, method='ward')
        num_clusters = fcluster(Z, t=distance_thresholds[level], criterion='distance').max()
        clusters = fcluster(Z, t=distance_thresholds[level], criterion='distance')
        print(f'\n\n***level {level}, clusters {num_clusters}')
        # Iterate over subclusters
        print(f'****** clustering at level {level} for parent {parent_node}')
        for cluster_id in range(1, num_clusters + 1):
            print(f'****** cluster_id {cluster_id}')
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_embeddings = parent_embeddings[cluster_indices]
            cluster_papers_df = parent_papers.iloc[cluster_indices]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            for i, paper in cluster_papers_df.iterrows():
                print(f" {paper['faiss_id']}, {paper['title']}")
        for cluster_id in range(1, num_clusters + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_embeddings = parent_embeddings[cluster_indices]
            cluster_papers_df = parent_papers.iloc[cluster_indices]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)

            #for i, paper in cluster_papers_df.iterrows():
            #    print(f" level {level} cluster {cluster_id}, id {paper['faiss_id']} {paper['title'][:32']}")
            # Create a new cluster node
            cluster_node = ClusterNode(node_id=len(cluster_nodes[level + 1]), parent_id=parent_node.node_id,
                                       level=level + 1, summary=None, embedding=cluster_embedding)

            # Add the cluster node to the dictionary
            cluster_nodes[level + 1][cluster_node.node_id] = cluster_node

            # Add the cluster node as a child of its parent node
            parent_node.children.append(cluster_node.node_id)

            # Add paper IDs to the cluster node if it's a leaf node
            if level + 1 >= num_levels or len(cluster_papers_df)>2:
                cluster_node.paper_ids = papers.iloc[cluster_indices]['faiss_id'].tolist()
                # Recursively split the subcluster
            else:
                print(f'\n recursive call to level {level+1}')
                split_cluster(cluster_node, cluster_embeddings, cluster_papers_df, level + 1)
        print(f'\n\n\nsplit_cluster return')
        
    # Initialize the root node of the lattice
    root_node = ClusterNode(node_id=0, parent_id=None, level=0, summary=None, embedding=None)

    # Initialize a dictionary to store cluster nodes at each level
    cluster_nodes = defaultdict(dict)
    cluster_nodes[0][0] = root_node

    # Start the depth-first cluster splitting from the root node
    split_cluster(root_node, embeddings, papers, 0)


        # Generate summaries for leaf nodes
    for node in cluster_nodes[num_levels].values():
        node.summary = generate_node_summary(node, cluster_nodes, papers)

    # Generate summaries for internal nodes (bottom-up)
    for level in range(num_levels - 1, 0, -1):
        for node in cluster_nodes[level].values():
            node.summary = generate_node_summary(node, cluster_nodes, papers)

    # Generate summary for the root node
    root_node.summary = generate_node_summary(root_node, cluster_nodes, papers)

    # Generate embeddings for internal nodes (bottom-up)
    for level in range(num_levels - 1, 0, -1):
        for node in cluster_nodes[level].values():
            child_embeddings = [cluster_nodes[level + 1][child_id].embedding for child_id in node.children]
            node.embedding = np.mean(child_embeddings, axis=0)

    # Generate embedding for the root node
    child_embeddings = [cluster_nodes[1][child_id].embedding for child_id in root_node.children]
    root_node.embedding = np.mean(child_embeddings, axis=0)

    return root_node, cluster_nodes


def save_lattice(root_node, cluster_nodes, filename):
    lattice_data = {
        'root_node': root_node,
        'cluster_nodes': cluster_nodes
    }
    with open(filename, 'wb') as file:
        pickle.dump(lattice_data, file)


def load_lattice(filename):
    with open(filename, 'rb') as file:
        lattice_data = pickle.load(file)
    return lattice_data['root_node'], lattice_data['cluster_nodes']


### higher level functionality for abstraction lattice

### first, integrate a new paper

def find_best_fitting_cluster(paper, root_node, cluster_nodes):
    # experiment - initialize similarity threshold to root
    paper_embedding = si.s2.embedding_request(paper['summary'], 'search_document: ')
    current_node = root_node
    similarity_threshold = cosine_similarity(np.array([paper_embedding]), np.array([root_node.embedding]))
    while current_node.children:
        max_similarity = -1
        best_child = None
        for child_id in current_node.children:
            child_node = cluster_nodes[current_node.level + 1][child_id]
            similarity = cosine_similarity(np.array([paper_embedding]), np.array([child_node.embedding]))
            if similarity > max_similarity:
                max_similarity = similarity
                best_child = child_node
        if max_similarity >= similarity_threshold:
            current_node = best_child
        else:
            break
    return current_node

def add_paper_to_cluster(paper, root_node, cluster_nodes, papers):
    current_node = root_node
    while current_node.children:
        max_similarity = -1
        best_child = None
        for child_id in current_node.children:
            child_node = cluster_nodes[current_node.level + 1][child_id]
            similarity = cosine_similarity(si.s2.embedding_request(paper['summary'], 'search_document: '), child_node.embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_child = child_node
        current_node = best_child

    # Add the paper to the leaf node
    current_node.paper_ids.append(paper['faiss_id'])
    current_node.summary = generate_node_summary(current_node, cluster_nodes, papers)
    current_node.embedding = np.mean([papers.loc[papers['paper_id'] == paper_id, 'embedding'].values[0] for paper_id in current_node.paper_ids], axis=0)

    # Update ancestor nodes
    update_ancestor_nodes(current_node, cluster_nodes, papers)

    # Split the leaf node if necessary
    split_cluster(current_node, cluster_nodes, papers, cluster_size_threshold)


def update_ancestor_nodes(node, cluster_nodes, papers):
    current_node = node
    while current_node.parent_id is not None:
        parent_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
        parent_node.summary = generate_node_summary(parent_node, cluster_nodes, papers)
        parent_node.embedding = np.mean([cluster_nodes[current_node.level][child_id].embedding for child_id in parent_node.children], axis=0)
        current_node = parent_node

def split_cluster(node, cluster_nodes, papers, cluster_size_threshold):
    if len(node.paper_ids) <= cluster_size_threshold:
        return
    
    subcluster_embeddings = [papers.loc[papers['paper_id'] == paper_id, 'embedding'].values[0] for paper_id in node.paper_ids]
    Z_subcluster = linkage(subcluster_embeddings, method='ward')
    num_subclusters = int(np.ceil(len(node.paper_ids) / cluster_size_threshold))
    subclusters = fcluster(Z_subcluster, num_subclusters, criterion='maxclust')
    
    node.children = []
    for subcluster_id in range(1, num_subclusters + 1):
        subcluster_indices = np.where(subclusters == subcluster_id)[0]
        subcluster_paper_ids = [node.paper_ids[i] for i in subcluster_indices]
        subcluster_embeddings = [subcluster_embeddings[i] for i in subcluster_indices]
        subcluster_embedding = np.mean(subcluster_embeddings, axis=0)
        
        subcluster_node = ClusterNode(node_id=len(cluster_nodes[node.level + 1]),
                                      parent_id=node.node_id,
                                      level=node.level + 1,
                                      summary=None,
                                      embedding=subcluster_embedding,
                                      paper_ids=subcluster_paper_ids)
        
        cluster_nodes[node.level + 1][subcluster_node.node_id] = subcluster_node
        node.children.append(subcluster_node.node_id)
    
    node.paper_ids = []
    node.summary = generate_node_summary(node, cluster_nodes, papers)
    node.embedding = np.mean([cluster_nodes[node.level + 1][child_id].embedding for child_id in node.children], axis=0)

def generate_integration_report(paper, cluster_node):
    local_summary = generate_local_contribution_summary(paper, cluster_node)
    broader_summaries = []
    current_node = cluster_node
    while current_node.parent_id is not None:
        parent_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
        broader_summary = generate_broader_contribution_summary(paper, parent_node, current_node.level - 1)
        broader_summaries.append(broader_summary)
        current_node = parent_node
    
    report = f"Integration Report for Paper {paper['faiss_id']}:\n\n"
    report += f"The paper has been integrated into the cluster lattice at level {cluster_node.level}, cluster {cluster_node.node_id}.\n\n"
    report += "Local Contributions:\n" + local_summary + "\n\n"
    report += "Broader Contributions:\n"
    for i, summary in enumerate(broader_summaries[::-1]):
        report += f"Level {i+1}:\n{summary}\n\n"
    
    return report


def generate_local_contribution_summary(paper, cluster_node):
    si.process2(arg1=paper['summary'],
                arg2=cluster_node.summary,
                instruction="""Compare Text1 to Text2, the current knowledge cluster Text1 is a member of.
Succinctly identify:
1. Those additional topics this paper covers, if any. 
2. Those new methods this paper introduces, if any.
3. Any new data this paper presents 
4. Any novel inferences or claims this paper makes
5. Any new conclusions this paper draws.
Conclude with an analysis of any significant shifts Text1 makes in the overall state of knowledge represented by the cluster.""",
                dest='$localContributionSummary',
                max_tokens = 600
                )
    return si.wm.get('$localContributionSummary')['item']


def generate_broader_contribution_summary(paper, node, level):
    si.process2(arg1=paper['summary'],
                arg2=node.summary,
                instruction=f"""Compare Text1 to Text2, the current abstract knowledge cluster Text1 is a member of.
Describe, at a high level appropriate for the {str(level)}th level of abstraction, 
how this paper, if at all, extends, complements, or contradicts the existing literature on this topic 
as represented by Text2, known fact, and logical reasoning.""",
                dest='$broaderContributionSummary_' + str(level),
                max_tokens = 600
                )
    return si.wm.get('$broaderContributionSummary_' + str(level))['item']

def generate_comprehensive_answer(paper, root, cluster_nodes):
    local_node = find_best_fitting_cluster(paper, root, cluster_nodes)
    local_summary = generate_local_contribution_summary(paper, local_node)
    broader_summaries = []
    current_node = local_node
    while current_node.parent_id is not None:
        parent_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
        broader_summary = generate_broader_contribution_summary(paper, parent_node, current_node.level - 1)
        broader_summaries.append(broader_summary)
        current_node = parent_node
    
    comprehensive_answer = 'Local contributions:\n' + local_summary + '\n\nBroader contributions:\n'
    for i, summary in enumerate(broader_summaries[::-1]):
        comprehensive_answer += f'Level {i+1}:\n{summary}\n\n'
    
    si.process1(comprehensive_answer,
                instruction='rewrite into a single coherent narrative',
                dest = '$review',
                max_tokens = 600)
    return si.wm.get('$review')['item']
                                
"""
# Example usage
papers = [
    {'paper_id': 1, 'summary': 'Paper 1 summary', 'embedding_vector': np.random.rand(512)},
    {'paper_id': 2, 'summary': 'Paper 2 summary', 'embedding_vector': np.random.rand(512)},
    # ... add more papers
]

num_levels = 3
cluster_size_thresholds = [10, 5]  # Adjust these thresholds based on your requirements

root_node, cluster_nodes = build_cluster_lattice(papers, num_levels, cluster_size_thresholds, None)

# Access the cluster nodes and their attributes
for level, nodes in cluster_nodes.items():
    print(f"Level {level}:")
    for node_id, node in nodes.items():
        print(f"Node ID: {node.node_id}")
        print(f"Parent ID: {node.parent_id}")
        print(f"Summary: {node.summary}")
        print(f"Embedding: {node.embedding}")
        print(f"Children: {node.children}")
        print(f"Paper IDs: {node.paper_ids}")
        print("---")

"""

#
### Browse lattice
#

import pickle

def load_lattice(filename):
    with open(filename, 'rb') as file:
        lattice_data = pickle.load(file)
    return lattice_data['root_node'], lattice_data['cluster_nodes']

def display_node_info(node):
    print(f"Node ID: {node.node_id}")
    print(f"Level: {node.level}")
    print(f"Parent ID: {node.parent_id}")
    print(f"Children IDs: {node.children}")
    print(f"Summary: {node.summary}")
    #print(f"Embedding: {node.embedding}")
    print(f"Paper IDs: {node.paper_ids}")
    print()

def browse_lattice(root_node, cluster_nodes):
    current_node = root_node
    while True:
        display_node_info(current_node)
        
        if current_node.children:
            print("Available actions:")
            print("1. Go to child node")
            print("2. Go to parent node")
            print("3. Quit")
            action = input("Enter the action number: ")
            
            if action == "1":
                child_id = int(input("Enter the child node ID: "))
                if child_id in current_node.children:
                    current_node = cluster_nodes[current_node.level + 1][child_id]
                else:
                    print("Invalid child node ID.")
            elif action == "2":
                if current_node.parent_id is not None:
                    current_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
                else:
                    print("Already at the root node.")
            elif action == "3":
                break
            else:
                print("Invalid action.")
        else:
            print("Available actions:")
            print("1. Go to parent node")
            print("2. Quit")
            action = input("Enter the action number: ")
            
            if action == "1":
                if current_node.parent_id is not None:
                    current_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
                else:
                    print("Already at the root node.")
            elif action == "2":
                break
            else:
                print("Invalid action.")



class LLMScript:
    def __init__ (self, interpreter, cot):
        self.interpreter = interpreter
        self.cot = cot
        self.s2 = s2
        s2.cot = cot
        rw.cot = cot
        self.wm = interpreter.wm
        print(f'\ncreated script runner!')
        
    def fetch(self, paper_id=None, uri=None, instruction='', dest=None, max_tokens=200):
        if paper_id is None:
            if uri is None:
                raise Error('need paper_id or uri!')
                paper_id = s2.index_url(uri)
        if paper_id < 0:
            raise ValueError('invalid paper_id or uri!')
        texts = s2.get_paper_sections(paper_id = paper_id)
        resource = rw.shorten(texts, instruction, max_tokens)
        self.wm.assign(dest, resource)
        return paper_id

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
        resolved_inst = instruction.replace(arg1, 'the user Input provided below')
        resolved_inst = resolved_inst.replace(arg2, 'the user SecondInput provided below')
        prompt = [SystemMessage("""{{$resolved_inst}}
Do not include any discursive or explanatory text. Limit your response to {{$length}} words.
End your response with </Response>"""),
                  UserMessage('\n<Text1>\n{{$text1}}\n</Text1>\n<Text2>\n{{$text2}}\n</Text2>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"resolved_inst":resolved_inst, "text1":resolved_arg1, "text2":resolved_arg2,
                                  "length":int(max_tokens/1.33)}, prompt, max_tokens=int(1.2*max_tokens), eos='</Response>')
        self.wm.assign(dest, query)
        return query
    

    def create_paper_summary(self, paper_id):
        paper_id = self.fetch(paper_id=paper_id,
                        instruction='extract the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                        dest='$paper1',
                        max_tokens=900
                        )
        #print(f'fetched paper_id {paper_id}')
        if paper_id:
            response = self.process1(arg1='$paper1', instruction='rewrite the content provided below as an integrated overview. Include all important details, but remove redundant information where possible.',
                                     dest='$summary',
                                     max_tokens=600
                                     )
            self.s2.set_paper_field(paper_id, 'summary', si.wm.get('$summary'))
            self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$summary", "result":'$trash'}])
            

    def create_paper_extract(self, paper_id):
        paper_id = self.fetch(paper_id=paper_id,
                        instruction='extract the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                        dest='$paper1',
                        max_tokens=2000
                        )
        #print(f'fetched paper_id {paper_id}')
        if paper_id:
            
            self.s2.set_paper_field(paper_id, 'extract', si.wm.get('$paper1'))

    def pre_process_paper(self, uri):
        ### extract and summarize a paper, add extract and summary to df
        # first call fetch, which creates full extract
        paper_id = self.s2.index_url(uri)
        if paper_id is not None:
            self.create_paper_extract(paper_id)
            self.create_paper_summary(paper_id)
            # lattice add paper - tbd
        self.s2.save_paper_df()
        self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$paper1Summary", "result":'$trash'}])
        

    def update_extracts(self):
        #sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in self.s2.paper_library_df.iterrows():
            #if row['summary'] is None or len(row['summary']) < 24:
            if row['extract'] is None or len(row['extract']) < 240:
                self.pre_process_paper(uri=row['pdf_url'])
            self.wm.assign('$paper', row['extract'])
            

    def update_summaries(self):
        #sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in self.s2.paper_library_df.iterrows():
            paper_id = row['faiss_id']
            self.create_paper_summary(paper_id)
            
        self.s2.save_paper_df()
            


        
if __name__=='__main__':
    cot = OwlCoT.OwlInnerVoice()
    print('created cot')
    interp = Interpreter.Interpreter(cot)
    print('created interp')
    si = LLMScript(interp, cot)
    print('created si')
    #si.update_summaries()
    paper = si.s2.get_paper_pd(id=46039551)
    #print(f'paper\n{paper}')
    root_node, nodes = build_cluster_lattice(si.s2.paper_library_df, 3, [1.4,1.2,1.0])
    save_lattice(root_node, nodes, 'lattice.pkl')
    root_node, nodes = load_lattice('lattice.pkl')
    #print(root_node, nodes)
    # print(generate_comprehensive_answer(paper, root_node, nodes))
    # Load the lattice from a file
    #root_node, nodes = load_lattice('lattice.pkl')

    # Start browsing the lattice
    browse_lattice(root_node, nodes)



"""
split_cluster return
generate node summary for 0, end=" of "
 node 0 of 0
including [323529736, 12218601, 227529427, 4183600, 313811026, 186147248, 171137813]
assign $paperSummary, <class 'str'>, This collection represents resea
generate node summary for 1, end=" of "
 node 1 of 1
including [323529736, 4183600, 186147248, 243597710, 41595642, 92387789, 248324031, 206591833]
assign $paperSummary, <class 'str'>, This collection represents resea
generate node summary for 2, end=" of "
 node 2 of 1
including [12218601, 227529427, 313811026, 171137813, 251801157, 164319949, 172297354, 138421729, 164430366, 297017174, 59827999]
"""
