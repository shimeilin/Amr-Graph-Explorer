# graph_query.py - Gene-centric Graph Query Engine

import pandas as pd
import networkx as nx
from rapidfuzz import process

# Paths to your graph CSVs
BASE = "/nobackup/hkkq91/ai_genomes/csv"
NODES_ARG = f"{BASE}/nodes_arg.csv"
NODES_CONTIG = f"{BASE}/nodes_contig.csv"
NODES_PLASMID = f"{BASE}/nodes_plasmid.csv"
NODES_SAMPLE = f"{BASE}/nodes_sample.csv"
EDGES_ARG = f"{BASE}/edges_contig_arg.csv"
EDGES_PLASMID = f"{BASE}/edges_contig_plasmid.csv"
EDGES_SAMPLE = f"{BASE}/edges_sample_contig.csv"

# Load all nodes and edges
def load_graph():
    nodes = pd.concat([
        pd.read_csv(NODES_ARG).assign(type='arg'),
        pd.read_csv(NODES_CONTIG).assign(type='contig'),
        pd.read_csv(NODES_PLASMID).assign(type='plasmid'),
        pd.read_csv(NODES_SAMPLE).assign(type='sample')
    ])
    edges = pd.concat([
        pd.read_csv(EDGES_ARG),
        pd.read_csv(EDGES_PLASMID),
        pd.read_csv(EDGES_SAMPLE)
    ])
    return nodes, edges

# Build graph using NetworkX
def build_nx_graph(nodes_df, edges_df):
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['id'], type=row['type'])
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], label=row.get('label', ''))
    return G

# Extract best-matching gene name from question
def extract_gene_alias(question: str, gene_list: list[str]):
    match, score, _ = process.extractOne(question, gene_list)
    return match if score > 60 else None

# Query function: given question → return graph result
def query_by_gene_name(question: str):
    nodes_df, edges_df = load_graph()
    G = build_nx_graph(nodes_df, edges_df)

    gene_list = nodes_df[nodes_df['type'] == 'arg']['id'].tolist()
    gene_match = extract_gene_alias(question, gene_list)
    if not gene_match:
        raise ValueError("Gene not found in database.")

    # Subgraph around that gene
    sub_nodes = set([gene_match])
    sub_nodes.update(nx.node_connected_component(G, gene_match))
    subgraph = G.subgraph(sub_nodes)

    result = {
        "args": [],
        "contigs": [],
        "samples": [],
        "plasmids": [],
        "graph": {
            "nodes": [],
            "edges": []
        }
    }

    for node_id in subgraph.nodes():
        node_type = G.nodes[node_id].get("type", "")
        if node_type in result:
            result[node_type + "s"].append(node_id)
        result["graph"]["nodes"].append({"id": node_id, "type": node_type})

    for source, target in subgraph.edges():
        result["graph"]["edges"].append({"source": source, "target": target})

    return result
