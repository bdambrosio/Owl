import os, sys
import json
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
import semanticScholar3 as s2
import rewrite as rw
import OwlCoT
import Interpreter
from LLMScript import LLMScript
import numpy as np
import pickle
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
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
import re

keys = ['key1', 'key2', 'key3']
result_string = '''key1: value1 with : and \n
key2: value2 with : and \n
key3: value3 with : and \n'''

def recover_dict_from_text(text, dimensions):
    # summary and excerpt texts in papers are dicts that have been rendered as flat lists.
    # this attempts to recover them
    if not text.startswith('\n'):
        text = '\n'+text
    keys = list(dimensions.keys())
    recovered_dict = {}
    for i in range(len(keys)):
        key = keys[i]
        if i == len(keys) - 1:
            # Last key, capture until the end of the string
            value = re.findall(f"\n{key}\n(.*)", text, re.DOTALL)[0].strip()
        else:
            # Capture until the next key
            next_key = keys[i + 1]
            value = re.findall(f"\n{key}\n(.*?)\n{next_key}\n", text, re.DOTALL)[0].strip()
        recovered_dict[key] = value
    return recovered_dict

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


def build_clusterNode_embeddings(num_levels, root_node, cluster_nodes):
    # Generate embeddings for internal nodes (bottom-up)
    for level in range(num_levels - 1, 0, -1):
        for node in cluster_nodes[level].values():
            if node.children is None or len(node.children) == 0:
                # leaf node embeddings are computed from papers!
                if np.any(np.isnan(node.embedding)):
                    raise ValueError(f'Leaf Node embedding is Nan: level {level}, id {node.node_id}\n{node.children}\n{node.paper_ids}')
                continue
            print(f'embedding level {level}, node {node.node_id}')
            child_embeddings = [cluster_nodes[level + 1][child_id].embedding for child_id in node.children if np.any(np.isnan(child_id.embedding)) is False]
            node.embedding = np.mean(child_embeddings, axis=0)
            if np.any(np.isnan(node.embedding)):
                raise ValueError(f'Node embedding is Nan: level {level}, id {node.node_id}\n{node.children}\n{node.paper_ids}')

    # Generate embedding for the root node
    child_embeddings = [cluster_nodes[1][child_id].embedding for child_id in root_node.children]
    root_node.embedding = np.mean(child_embeddings, axis=0)
    if np.any(np.isnan(root_node.embedding)):
        raise ValueError(f'Root Node embedding is Nan!')

def generate_node_summary(node, cluster_nodes, papers, max_children=20):
    global si
    if node.paper_ids:
        print(f'generate node summary for {node.node_id} of {node.parent_id} including {node.paper_ids}')
        for paper_id in node.paper_ids:
            print(f"  {papers.loc[papers['faiss_id'] == paper_id, 'title'].iloc[0][:64]}")
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
            cluster_inds = np.where(clusters == cluster_id)[0]
            cluster_papers_df = parent_papers.iloc[cluster_inds]
            for i, paper in cluster_papers_df.iterrows():
                print(f" {paper['faiss_id']}, {paper['title']}")
        for cluster_id in range(1, num_clusters + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_embeddings = parent_embeddings[cluster_indices]
            cluster_papers_df = parent_papers.iloc[cluster_indices]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            if np.any(np.isnan(cluster_embedding)):
                raise ValueError(f'cluster embedding is Nan: level {level}, cluster_id {cluster_id}')

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

    build_clusterNode_embeddings(num_levels, root_node, cluster_nodes)

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

def find_best_fitting_cluster(paper, root_node, cluster_nodes, paper_embedding = None):
    # experiment - initialize similarity threshold to root
    if paper_embedding is None:
        paper_embedding = si.s2.embedding_request(paper['summary'], 'search_document: ')
    current_node = root_node
    if np.any(np.isnan(paper_embedding)):
        print (f"Paper embedding is Nan!")
    if np.any(np.isnan(root_node.embedding)):
        print (f"Root node embedding is Nan!")
    try:
        similarity_threshold = cosine_similarity(np.array([paper_embedding]), np.array([root_node.embedding]))
    except Exception as e:
        print (f"str(e)\n{paper['faiss_id']}, {type(paper_embedding)}, {type(root_node.embedding)}")
        raise e
    while current_node.children:
        max_similarity = -1
        best_child = None
        for child_id in current_node.children:
            child_node = cluster_nodes[current_node.level + 1][child_id]
            if np.any(np.isnan(child_node.embedding)):
                print (f"child node embedding is Nan!")
            try:
                similarity = cosine_similarity(np.array([paper_embedding]), np.array([child_node.embedding]))
            except Exception as e:
                print (f"str(e)\n{paper['faiss_id']}, {type(paper_embedding)}, {type(child_node.embedding)}")
                raise e
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
    print('\n\n\nGENERATE LOCAL CONTRIBUTION\n')
    si.process2(arg1=paper['summary'],
                arg2=cluster_node.summary,
                instruction="""Compare Text1 to Text2. Text2 is the knowledge cluster Text1 is most closely related to.
Succinctly identify:
1. Those new topics this paper covers, if any. 
2. Those new methods this paper introduces, if any.
3. Any new data this paper presents. 
4. Any novel inferences or claims this paper makes.
5. Any new conclusions this paper draws.
Conclude with an analysis of any significant shifts Text1 makes in the overall state of knowledge represented by Text2.""",
                dest='$localContribution',
                max_tokens = 600
                )
    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$localContribution", "result":'$trash'}])
    return si.wm.get('$localContribution')['item']


def generate_broader_contribution_summary(paper, node, level):
    print('\n\n\nGENERATE BROADER CONTRIBUTION\n')
    si.process2(arg1=paper['summary'],
                arg2=node.summary,
                instruction=f"""Compare Text1 to Text2, an abstract knowledge cluster Text1 is most closely related to.
Describe, at a high level appropriate for the {str(level)}th level of abstraction, 
how this paper, if at all, extends, complements, or contradicts the existing literature on this topic 
as represented by Text2, known fact, and logical reasoning.""",
                dest='$broaderContribution'+str(level),
                max_tokens = 600
                )
    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$broaderContribution"+str(level), "result":'$trash'}])
    return si.wm.get('$broaderContribution'+str(level))['item']

def generate_comprehensive_answer(paper, root, cluster_nodes):
    print('\n\n\nGENERATE COMPREHENSIVE CONTRIBUTION\n')
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
    print(f"Embedding: {type(node.embedding)}")
    print(f"Paper IDs: {node.paper_ids}")
    print()

def browse_lattice(root_node, cluster_nodes):
    current_node = root_node
    while True:
        display_node_info(current_node)
        
        if current_node.children:
            print("Available actions:")
            print("1. Display paper")
            print("2. Go to child node")
            print("3. Go to parent node")
            print("4. Quit")
            action = input("Enter the action number: ")
            
            if action == "1":
                paper_id = int(input("Enter the paper faiss id: "))
                paper = si.s2.get_paper_pd(paper_id=int(paper_id))
                if paper is not None:
                    print(paper['summary'])
                else:
                    print('cant find paper id {paper_id}')
            elif action == "2":
                child_id = int(input("Enter the child node ID: "))
                if child_id in current_node.children:
                    current_node = cluster_nodes[current_node.level + 1][child_id]
                else:
                    print("Invalid child node ID.")
            elif action == "3":
                if current_node.parent_id is not None:
                    current_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
                else:
                    print("Already at the root node.")
            elif action == "4":
                break
            else:
                print("Invalid action.")
        else:
            print("Available actions:")
            print("1. Display paper")
            print("2. Go to parent node")
            print("3. Quit")
            action = input("Enter the action number: ")
            if action == '1':
                paper = si.s2.get_paper_pd(paper_id=int(paper_id))
                if paper is not None:
                    print(paper['summary'])
                else:
                    print('cant find paper id {paper_id}')
            elif action == "2":
                if current_node.parent_id is not None:
                    current_node = cluster_nodes[current_node.level - 1][current_node.parent_id]
                else:
                    print("Already at the root node.")
            elif action == "3":
                break
            else:
                print("Invalid action.")


if __name__=='__main__':
    cot = OwlCoT.OwlInnerVoice()
    print('created cot')
    interp = Interpreter.Interpreter(cot)
    print('created interp')
    si = LLMScript(interp, cot)
    print('created si')
    root_node, nodes = build_cluster_lattice(si.s2.paper_library_df, 3, [1.4,1.2,1.0])
    save_lattice(root_node, nodes, 'lattice.pkl')
    root_node, nodes = load_lattice('lattice.pkl')
    #si.create_paper_summary(paper_id=12218601)
    #si.create_paper_extract(paper_id=12218601)
    #save_lattice(root_node, nodes, 'lattice.pkl')
    #si.update_extracts()
    #si.s2.save_paper_df()
    paper = si.s2.get_paper_pd(paper_id=12218601)
    summary_dict = recover_dict_from_text(paper['summary'], rw.default_dimensions)
    #print(json.dumps(summary_dict, indent=2))
    #si.update_extracts()
    #save_lattice(root_node, nodes, 'lattice.pkl')
    #print(root_node, nodes)
    #save_lattice(root_node, nodes, 'lattice.pkl')
    #paper = si.s2.get_paper_pd(paper_id=93853867)
    #paper = si.s2.get_paper_pd(paper_id=12218601)
    #print(generate_comprehensive_answer(paper, root_node, nodes))
    # Load the lattice from a file
    #root_node, nodes = load_lattice('lattice.pkl')
    #si.create_paper_novelty(93853867)
    # Start browsing the lattice
    #build_clusterNode_embeddings(3, root_node, nodes)
    #browse_lattice(root_node, nodes)
