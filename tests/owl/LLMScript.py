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

def generate_node_summary(node, cluster_nodes, papers, max_children=20):
    global si
    if node.paper_ids:
        # If the node has paper IDs, it's a leaf node
        summaries = [papers.loc[papers['faiss_id'] == paper_id, 'summary'].iloc[0] for paper_id in node.paper_ids]
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
                  instruction='place this collection in the context of the broader field. Then provide a overall summary of the topics or problem addressed, methods used, data presented, inferences or claims made, and conclusions drawn. Do not comment on each paragraph or paper individually. Rather, your response should represent the collection as a whole and the outstanding contributions of the collection.',
                  dest='$paperSummary',
                  max_tokens=500
                  )
    return si.wm.get('$paperSummary')['item']


def build_cluster_lattice(papers, num_levels, cluster_size_thresholds):
    # Extract embeddings from papers
    embeddings = np.array([si.s2.embedding_request(paper, 'search_document: ') for paper in papers['summary']])

    # Perform hierarchical clustering
    Z = linkage(embeddings, method='ward')

    # Initialize the root node of the lattice
    root_node = ClusterNode(node_id=0, parent_id=None, level=0, summary=None, embedding=None)

    # Initialize a dictionary to store cluster nodes at each level
    cluster_nodes = defaultdict(dict)
    cluster_nodes[0][0] = root_node

    # Iterate over levels
    for level in range(1, num_levels + 1):
        # Determine the number of clusters at the current level
        num_clusters = len(cluster_nodes[level - 1])

        # Assign cluster labels for the current level
        clusters = fcluster(Z, num_clusters, criterion='maxclust')

        # Iterate over clusters at the current level
        for cluster_id in range(1, num_clusters + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)

            # Create a new cluster node
            cluster_node = ClusterNode(node_id=len(cluster_nodes[level]), parent_id=cluster_id - 1, level=level,
                                       summary=None, embedding=cluster_embedding)

            # Add paper IDs to the cluster node if it's a leaf node
            if level == num_levels:
                cluster_node.paper_ids = papers.iloc[cluster_indices]['faiss_id'].tolist()

            # Add the cluster node to the dictionary
            cluster_nodes[level][cluster_node.node_id] = cluster_node

            # Add the cluster node as a child of its parent node
            parent_node = cluster_nodes[level - 1][cluster_node.parent_id]
            parent_node.children.append(cluster_node.node_id)

        # Check if any clusters need to be split into subclusters
        if level < num_levels:
            cluster_size_threshold = cluster_size_thresholds[level - 1]
            large_clusters = [c for c in cluster_nodes[level].values() if len(c.paper_ids) > cluster_size_threshold]

            for large_cluster in large_clusters:
                subcluster_indices = np.where(clusters == large_cluster.node_id + 1)[0]
                subcluster_embeddings = embeddings[subcluster_indices]
                Z_subcluster = linkage(subcluster_embeddings, method='ward')
                num_subclusters = int(np.ceil(len(subcluster_indices) / cluster_size_threshold))
                subclusters = fcluster(Z_subcluster, num_subclusters, criterion='maxclust')

                for subcluster_id in range(1, num_subclusters + 1):
                    subcluster_indices_relative = np.where(subclusters == subcluster_id)[0]
                    subcluster_indices_absolute = subcluster_indices[subcluster_indices_relative]
                    subcluster_embeddings = embeddings[subcluster_indices_absolute]
                    subcluster_embedding = np.mean(subcluster_embeddings, axis=0)

                    # Create a new subcluster node
                    subcluster_node = ClusterNode(node_id=len(cluster_nodes[level + 1]), parent_id=large_cluster.node_id,
                                                  level=level + 1, summary=None, embedding=subcluster_embedding,
                                                  paper_ids=papers.iloc[subcluster_indices_absolute]['paper_id'].tolist())

                    # Add the subcluster node to the dictionary
                    cluster_nodes[level + 1][subcluster_node.node_id] = subcluster_node

                    # Add the subcluster node as a child of its parent node
                    large_cluster.children.append(subcluster_node.node_id)

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

def add_paper_to_cluster(paper, cluster_node, cluster_nodes, papers):
    if cluster_node.level == len(cluster_nodes) - 1:  # Leaf node
        cluster_node.paper_ids.append(paper['faiss_id'])
    else:  # Internal node
        new_child_node = ClusterNode(node_id=len(cluster_nodes[cluster_node.level + 1]),
                                     parent_id=cluster_node.node_id,
                                     level=cluster_node.level + 1,
                                     summary=None,
                                     embedding= si.s2.embedding_request(paper['summary'], 'search_document: '),
                                     paper_ids=[paper['faiss_id']])
        cluster_nodes[cluster_node.level + 1][new_child_node.node_id] = new_child_node
        cluster_node.children.append(new_child_node.node_id)
    
    cluster_node.summary = generate_node_summary(cluster_node, cluster_nodes, papers)
    cluster_node.embedding = np.mean([papers.loc[papers['paper_id'] == paper_id, 'embedding'].values[0] for paper_id in cluster_node.paper_ids], axis=0)

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
    si.process1(arg1=paper['summary'],
                instruction='compare this paper to the following summary which represents the current knowledge cluster it belongs to: ' + cluster_node.summary +\
""" and then succinctly identify:
1. what if any additional topics this paper covers 
2. what if any new methods this paper introduces
3. what if any new data this paper presents 
4. what if any novel inferences or claims this paper makes
5. what if any new conclusions this paper draws""",
                dest='$localContributionSummary',
                max_tokens = 600
                )
    return si.wm.get('$localContributionSummary')['item']

def generate_broader_contribution_summary(paper, node, level):
    si.process1(arg1=paper['summary'],
                instruction=f"""compare this paper to the following higher-level summary of the literature:

<Summary>
{node.summary}
</Summary>

and then succinctly describe, at a high level appropriate for the {str(level)}th level of abstraction, 
how this paper, if at all, extends, complements, or contradicts the existing literature on this topic 
in terms of the broad topics, approaches, evidence and conclusions covered.""",
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
    
    return comprehensive_answer
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


class LLMScript:
    def __init__ (self, interpreter, cot):
        self.interpreter = interpreter
        self.cot = cot
        self.s2 = s2
        s2.cot = cot
        rw.cot = cot
        self.wm = interpreter.wm
        print(f'\ncreated script runner!')
        
    def fetch(self, uri, instruction, dest, max_tokens):
        paper_id = s2.index_url(uri)
        if paper_id < 0:
            return False
        texts = s2.get_paper_sections(paper_id = paper_id)
        resource = rw.shorten(texts, instruction, max_tokens)
        self.wm.assign(dest, resource)
        return paper_id

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process1(self, arg1, instruction, dest, max_tokens):
        resolved_arg= self.interpreter.resolve_arg(arg1)
        resolved_inst = instruction.replace(arg1, 'the user input provided below')
        prompt = [SystemMessage("""{{$resolved_inst}}
Do not include any discursive or explanatory text. End your response with </Response>"""),
                  UserMessage('\n<INPUT>\n{{$input}}\n</INPUT>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"resolved_inst":resolved_inst, "input":resolved_arg, "length":int(max_tokens)},
                            prompt, max_tokens=int(1.2*max_tokens), eos='</Query>')
        self.wm.assign(dest, query)
        return query

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process2(self, arg1, arg2, instruction, dest, max_tokens):
        resolved_arg1 = self.interpreter.resolve_arg(arg1)
        resolved_arg2 = self.interpreter.resolve_arg(arg2)
        resolved_inst = instruction.replace(arg1, 'the user Input provided below')
        resolved_inst = instruction.replace(arg2, 'the user SecondInput provided below')
        prompt = [SystemMessage("""{{$resolved_inst}}
Do not include any discursive or explanatory text. End your response with </Query>"""),
                  UserMessage('\n<Input>\n{{$text1}}\n</Input>\n<SecondInput>\n{{$text2}}\n</SecondInput>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"resolved_inst":resolved_inst, "text1":resolved_arg1, "text2":resolved_arg2,
                                  "length":int(max_tokens)}, prompt, max_tokens=int(1.2*max_tokens), eos='</Response>')
        self.wm.assign(dest, query)
        return query
    


    def pre_process_paper(self, uri):
        ### extract and summarize a paper, add extract and summary to df
        # first call fetch, which creates full extract
        paper_id = self.fetch(uri=uri,
                        instruction='extract the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                        dest='$paper1',
                        max_tokens=4000
                        )
        #print(f'fetched paper_id {paper_id}')
        if paper_id:
            self.s2.set_paper_field(paper_id, 'extract', si.wm.get('$paper1'))

            # now call process1 to create short summary for clustering
            self.process1(arg1='$paper1',
                          instruction='place this paper in the context of the broader field. Then provide a summary of the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                          dest='$paper1Summary',
                          max_tokens=400
                          )
            self.s2.set_paper_field(paper_id, 'summary', si.wm.get('$paper1Summary'))
            self.s2.save_paper_df()
        
            self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$paper1Summary", "result":'$trash'}])
        

    def build_lattice(self):
        #
        ### designed and implemented by Claude3, with help from me.
        #
        embeddings = []
        titles = []
        for index, paper_pd in self.s2.paper_library_df.iterrows():
            if paper_pd is not None:
                paper_title = paper_pd['title']
                paper_summary = paper_pd['summary']
                paper_extract = paper_pd['extract']
                paper_embedding = si.s2.embedding_request(paper_summary, 'search_document: ')
                #print(f'{paper_id}\n{paper_summary}\n\n')
                embeddings.append(paper_embedding)
                titles.append(paper_title)
            
        embeddings = np.array(embeddings)
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from matplotlib import pyplot as plt

        # Assume you have a 300x512 numpy array called 'embeddings' containing the Specter2 embeddings
        # embeddings = ...
    
        # Perform hierarchical clustering
        # You can choose different linkage methods like 'ward', 'single', 'complete', 'average'
        Z = linkage(embeddings, method='ward')
    
        # Define the number of levels in the hierarchy and the number of clusters at each level
        num_levels = 3
        num_clusters_per_level = [5, 4, 3]  # Example: 5 clusters at level 1, 3 clusters at level 2, 2 clusters at level 3

        # Initialize a list to store the cluster assignments at each level
        cluster_assignments = []

        # Iterate over each level in the hierarchy
        for level in range(num_levels):
            # Determine the cut-off distance for the current level
            max_distance = Z[-num_clusters_per_level[level]+1, 2]
    
            # Assign cluster labels for the current level
            clusters = fcluster(Z, max_distance, criterion='distance')
            cluster_assignments.append(clusters)
    
            print(f"Level {level+1} - Number of clusters: {num_clusters_per_level[level]}")
            for cluster in range(1, num_clusters_per_level[level]+1):
                print(f"  Cluster {cluster} contains {np.sum(clusters == cluster)} papers")


        plt.figure(figsize=(10, 5))
        dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Paper Index')
        plt.ylabel('Distance')
        plt.show()
    
        # Choose a cutoff distance to define the clusters
        max_distance = 1.0  # Adjust this value based on your needs
        clusters = fcluster(Z, max_distance, criterion='distance')
        
        # Print the cluster assignments for each paper
        for i in range(len(clusters)):
            print(f"Paper {i+1} cluster {clusters[i]} {titles[i]}")
        
        # Count the number of papers in each cluster
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            count = np.sum(clusters == cluster)
            print(f"Cluster {cluster} contains {count} papers")

        sys.exit(0)
        # build summaries from bottom up for each cluster
        # Assume you have the cluster assignments from above code

        from collections import defaultdict
        # Generate cluster summaries
        cluster_summaries = defaultdict(lambda: defaultdict(list))
        for level in range(num_levels):
            for cluster in range(1, num_clusters_per_level[level]+1):
                cluster_papers = [papers[i] for i in range(len(papers)) if cluster_assignments[level][i] == cluster]
                summary = self.generate_cluster_summary(cluster_papers)  # Use the LLM to generate the summary
                cluster_summaries[level][cluster] = summary

        # Create cluster embeddings
        cluster_embeddings = defaultdict(lambda: defaultdict(list))
        for level in range(num_levels):
            for cluster in range(1, num_clusters_per_level[level]+1):
                cluster_paper_indices = [i for i in range(len(papers)) if cluster_assignments[level][i] == cluster]
                cluster_paper_embeddings = embeddings[cluster_paper_indices]
                centroid = np.mean(cluster_paper_embeddings, axis=0)
                cluster_embeddings[level][cluster] = centroid# Plot the dendrogram
            
    def update_summaries(self):
        sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in self.s2.paper_library_df.iterrows():
            #if row['summary'] is None or len(row['summary']) < 24:
            if row['extract'] is None or len(row['extract']) < 240:
                self.pre_process_paper(uri=row['pdf_url'])
            self.wm.assign('$paper', row['extract'])
            
            # call process1 to create short summary for clustering
            self.process1(arg1='$paper',
                          instruction='place this paper in the context of the broader field. Then provide a summary of the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                          dest='$paperSummary',
                          max_tokens=400
                          )
            paper_id = row['faiss_id']
            self.s2.set_paper_field(paper_id, 'summary', si.wm.get('$paperSummary'))
        self.s2.save_paper_df()
            
    
if __name__=='__main__':
    cot = OwlCoT.OwlInnerVoice()
    print('created cot')
    interp = Interpreter.Interpreter(cot)
    print('created interp')
    si = LLMScript(interp, cot)
    print('created si')
    paper = si.s2.get_paper_pd(id=46039551)
    #print(f'paper\n{paper}')
    #root, nodes = build_cluster_lattice(si.s2.paper_library_df, 3, [4,4,4])
    #save_lattice(root, nodes, 'cluster.pkl')
    root, nodes = load_lattice('cluster.pkl')
    #print(root, nodes)
    print(generate_comprehensive_answer(paper, root, nodes))


