#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph generation functions for log data
"""
import os
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import shutil
import json
from sentence_transformers import SentenceTransformer
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected

#from preprocessing.adjacency import get_adjacency_matrix
from preprocessing.adjacency import get_undirected_adj, get_appr_directed_adj, get_second_directed_adj

#from adjacency import get_undirected_adj, get_appr_directed_adj, get_second_directed_adj

# Global variable for root path - gets set in main() or when first accessed
ROOT_PATH = None

def get_root_path():
    """Get the root path for data files, initializing it if not already set"""

    ROOT_PATH = "/home/rcp_user/suzhang/graph/logade"
    return ROOT_PATH

def set_root_path(path):
    """Set the root path explicitly"""
    global ROOT_PATH
    if path:
        ROOT_PATH = path
        os.environ['LOGS2GRAPH_ROOT'] = path
        print(f"Set root path to {path}")

def get_dataset_path(dataset_name, subdir=None, filename=None):
    """Construct path for dataset files"""
    path = os.path.join(get_root_path(), "Data", dataset_name)
    if subdir:
        path = os.path.join(path, subdir)
    if filename:
        path = os.path.join(path, filename)
    return path

def load_dataset(dataset_name):
    """Load raw log dataset and generate embeddings using Sentence Transformer"""
    # Import sentence transformer (importing here to avoid dependency issues if not needed)
    
    # Dataset-specific loading logic
    if dataset_name == 'Kubelet':
        # Load log data
        df = pd.read_csv(get_dataset_path(dataset_name, filename="Kubelet.log_structured.csv"), sep=',')

        # Use PodUid column directly for group identification
        if 'PodUid' in df.columns:
            # Directly use PodUid column for GroupId
            df['GroupId'] = df['PodUid']
            print("Using PodUid column for group identification")
        else:
            # Raise an error if PodUid column is not available
            raise ValueError("PodUid column is required in the Kubelet.log_structured.csv file. Please ensure the data includes this column.")
                
        # Check if embeddings exist first before generating them
        embedding_path = get_dataset_path(dataset_name, filename="embeddings.json")
        if os.path.exists(embedding_path):
            print(f"Loading existing embeddings from {embedding_path}")
            with open(embedding_path, 'r') as fp:
                embedding_dict = json.load(fp)
        else:
            # Initialize Sentence Transformer model - using a model good for technical text
            print("Loading Sentence Transformer model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings for event templates
            print("Generating embeddings for log templates...")
            templates = df['EventTemplate'].unique()
            embeddings = model.encode(templates, show_progress_bar=True)
            
            # Create embedding dictionary
            embedding_dict = {}
            for template, embedding in zip(templates, embeddings):
                embedding_dict[template] = embedding.tolist()
            
            # Save embeddings for future use
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            with open(embedding_path, 'w') as fp:
                json.dump(embedding_dict, fp)
            print(f"Embeddings saved to {embedding_path}")        
    elif dataset_name == 'BGL':
        # Load log data
        df = pd.read_csv(get_dataset_path(dataset_name, filename="HDFS.log_structured.csv"), sep=',')
        df['GroupId'] = df['ParameterList'].str.extract('(blk\_[-]?\d+)', expand=False)
        
        # Check if embeddings exist, if not, generate them
        embedding_path = get_dataset_path(dataset_name, filename="embeddings.json")
        if os.path.exists(embedding_path):
            with open(embedding_path, 'r') as fp:
                embedding_dict = json.load(fp)
        else:
            # Initialize Sentence Transformer model
            print("Loading Sentence Transformer model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings for event templates
            print("Generating embeddings for log templates...")
            templates = df['EventTemplate'].unique()
            embeddings = model.encode(templates, show_progress_bar=True)
            
            # Create embedding dictionary
            embedding_dict = {}
            for template, embedding in zip(templates, embeddings):
                embedding_dict[template] = embedding.tolist()
            
            # Save embeddings for future use
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            with open(embedding_path, 'w') as fp:
                json.dump(embedding_dict, fp)
            print(f"Embeddings saved to {embedding_path}")
            
    # Add other datasets as needed...
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Extract relevant columns
    raw_df = df[["LineId", "LineId", "GroupId", "EventTemplate"]]
    embedding_df = pd.DataFrame.from_dict(embedding_dict, orient='index')
    
    # Create a node-to-log mapping for future reference
    node_mapping = {}
    for idx, group in raw_df.groupby('GroupId'):
        group_mapping = {}
        unique_events = group['EventTemplate'].unique()
        for i, event in enumerate(unique_events):
            event_logs = group[group['EventTemplate'] == event]
            group_mapping[str(i)] = {
                "template": event,
                "count": len(event_logs)
            }
        node_mapping[idx] = group_mapping
    
    # Save node mapping for use in anomaly inspection
    mapping_path = get_dataset_path(dataset_name, filename="node_to_log_mapping.json")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, 'w') as fp:
        json.dump(node_mapping, fp)
    
    return raw_df, embedding_df

def sample_groups(dataset_name, num_samples, anomaly_percentage):
    """Sample groups with controlled anomaly percentage"""
    # Load anomaly labels
    all_event_df = pd.read_csv(get_dataset_path(dataset_name, filename="anomaly_label.csv"), sep=',')
    group_to_check = list(all_event_df["PodUid"])
    
    # Get indices of normal and anomaly samples
    anomaly_indices = all_event_df.index[all_event_df['Label'] == "Anomaly"].tolist()
    normal_indices = all_event_df.index[all_event_df['Label'] == "Normal"].tolist()
    
    # Random sampling
    np.random.seed(42)  # For reproducibility
    
    if len(anomaly_indices) > 0 and len(normal_indices) > 0:
        # Calculate number of anomaly and normal samples
        num_anomaly = int(num_samples * anomaly_percentage)
        num_normal = num_samples - num_anomaly
        
        # Ensure we don't request more samples than available
        num_anomaly = min(num_anomaly, len(anomaly_indices))
        num_normal = min(num_normal, len(normal_indices))
        
        # Sample indices
        anomaly_samples = np.random.choice(anomaly_indices, num_anomaly, replace=False)
        normal_samples = np.random.choice(normal_indices, num_normal, replace=False)
        
        # Combine samples
        sampled_indices = np.concatenate([anomaly_samples, normal_samples])
        np.random.shuffle(sampled_indices)  # Shuffle to mix normal and anomaly samples
    else:
        # If no anomalies or no normal samples, just sample randomly
        sampled_indices = np.random.choice(len(group_to_check), num_samples, replace=False)
    
    # Get the corresponding group IDs
    sampled_groups = [group_to_check[i] for i in sampled_indices]
    
    return sampled_groups, sampled_indices

def read_file(folder, prefix, name, dtype=None):
    """Read a file in TU dataset format."""
    import os.path as osp
    
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

def construct_graph(log_df, group_name, embedding_df, dataset_name, graph_index, graph_location_index):
    """Construct a graph from log events for a specific group"""
    # Filter logs for the specific group
    group_df = log_df[log_df["GroupId"] == group_name]
    
    # If no logs for this group, return None
    if len(group_df) == 0:
        return None
    
    # Create directed graph
    G = nx.MultiDiGraph()
    
    # Extract event templates and add nodes
    event_list = list(group_df["EventTemplate"])
    unique_events = list(dict.fromkeys(event_list))
    G.add_nodes_from(unique_events)
    
    # Add edges between consecutive events
    if len(event_list) > 1:
        G.add_edges_from([(event_list[i], event_list[i+1]) for i in range(len(event_list)-1)])
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G)
    
    # Prepare output directory
    temp_dir = get_dataset_path(dataset_name, subdir="Graph/TempRaw")
    
    # Write adjacency matrix
    row_indices, col_indices = A.nonzero()
    edge_data = np.column_stack((row_indices + 1, col_indices + 1))  # +1 for 1-indexed
    np.savetxt(
        os.path.join(temp_dir, f"{dataset_name}_A.txt"),
        edge_data,
        fmt='%i',
        delimiter=', '
    )
    
    # Write edge weights
    edge_weights = pd.DataFrame({"edge_weight": list(A.data)})
    np.savetxt(
        os.path.join(temp_dir, f"{dataset_name}_edge_attributes.txt"),
        edge_weights.values,
        fmt='%i',
        delimiter=', '
    )
    
    # Write graph indicator
    graph_indicator = pd.DataFrame({"indicator": [graph_index + 1] * len(unique_events)})
    np.savetxt(
        os.path.join(temp_dir, f"{dataset_name}_graph_indicator.txt"),
        graph_indicator.values,
        fmt='%i',
        delimiter=', '
    )
    
    # Write graph labels (normal/anomaly)
    df_label = pd.read_csv(get_dataset_path(dataset_name, filename="anomaly_label.csv"), sep=',')
    df_label = df_label.replace({"Label": {"Normal": 0, "Anomaly": 1}})
    label_value = df_label.iloc[graph_location_index]['Label']
    
    graph_labels = pd.DataFrame({"labels": [label_value]})
    np.savetxt(
        os.path.join(temp_dir, f"{dataset_name}_graph_labels.txt"),
        graph_labels.values,
        fmt='%i',
        delimiter=', '
    )
    
    # Write node attributes (embeddings)
# Fix for line 260 in construct_graph function

    # Write node attributes (embeddings)
    node_attributes = []
    for event in unique_events:
        if event in embedding_df.index:
            # Get embedding directly as a list
            node_attributes.append(embedding_df.loc[event].tolist())
        else:
            # Fallback for missing embeddings
            # Determine embedding dimension from the first item in the dataframe
            if embedding_df.shape[0] > 0:
                # Get the first embedding and find its dimension
                first_embedding = embedding_df.iloc[0].tolist()
                if isinstance(first_embedding, list):
                    dim = len(first_embedding)
                else:
                    # If it's already a flat array, use the number of columns
                    dim = embedding_df.shape[1]
            else:
                # Default fallback dimension
                dim = 100
            
            print(f"Warning: No embedding found for event '{event}'. Using zero vector of dimension {dim}.")
            node_attributes.append([0.0] * dim)

    np.savetxt(
        os.path.join(temp_dir, f"{dataset_name}_node_attributes.txt"),
        np.array(node_attributes),
        fmt='%f',
        delimiter=', '
    )
    
    return G


def read_tu_data(folder, prefix, adj_type):
    
    # =============================================================================
    # read edge index from adj matrix
    # =============================================================================
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1 
    
    # print("\n-----edge_index in read_tu_data()-------")
    # print(edge_index)
    
    # =============================================================================
    # read graph index
    # =============================================================================    
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    # =============================================================================
    # read node attributes
    # =============================================================================
    
    if batch.dim() == 0: ## the batch looks like ->tensor(42), which is zero dimension
        node_attributes = torch.empty((1, 0))
        
    else: ## the batch looks like ->tensor([41, 41, 41, 41, 41, 41]), which is one dimension
        node_attributes = torch.empty((batch.size(0), 0))
    node_attributes = read_file(folder, prefix, 'node_attributes')
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    # =============================================================================
    # read edge attributes
    # =============================================================================
    # print("edge_index.shape")
    # print(edge_index.shape)
    
    is_empty_index = 0
    
    if len(edge_index.shape) == 1:##some graph only have a single row
        
        is_empty_index = 1
        data = Data()
        print("---we have empty graph here---")
        return data,is_empty_index  ##if it is empty, we skip this dataset
    
        if edge_index.shape[0] == 0:
          
            ##if it is empty, which means one node without any edges, we build a self-loop edge
            edge_index = torch.tensor([[1],[1]])
            
            is_empty_index = 1
            data = Data()
            print("---we have empty graph here---")
            return data,is_empty_index  ##if it is empty, we skip this dataset            
        else:           
            ##if this row is not empty, which mean two node with one edge
            edge_index = torch.tensor([[edge_index[0].item()],[edge_index[1].item()]])
        
    
    ##some graphs only have a single node, we should skip those graphs?
    
    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, prefix, 'edge_attributes')
    
    # print(edge_attributes)
    
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)


    # =============================================================================
    # concategate node attributes
    # =============================================================================
    # print("---------node-cat---------------")
    x = cat([node_attributes])
    
    # print("-----x.size(0)------")
    # print(x.size(0))
    
    # =============================================================================
    # concategate edge attributes and edge lables
    # =============================================================================
    # edge_attr = cat([edge_attributes, edge_labels])
    # print("---------edge-cat---------------")
    
    if edge_index.size(1) == 1: ##some graph only have a single row, this causes tensor with 0 dimension
        
        edge_attr = torch.tensor([[edge_attributes.item()]])
        
        # ##if it is empty, which means one node without any edges, we build a self-loop edge
        # if is_empty_index == 1:
            
        #     edge_attr = torch.tensor([[1]])
            
        # ##if this row is not empty, which mean two node with one edge
        # else:
        #     edge_attr = torch.tensor([[edge_attributes.item()]])
        
    else:       
        edge_attr = cat([edge_attributes])
        

    # =============================================================================
    # read graph attributes or graph labels
    # =============================================================================
    y = None
    y = read_file(folder, prefix, 'graph_labels', torch.long)

    # =============================================================================
    # get total number of nodes for all graphs
    # =============================================================================
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    # =============================================================================
    # remove self-loops: we should not remove selfloops
    # =============================================================================    
    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr) 
    
    if edge_attr is None:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    else:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
        
        
    # =============================================================================
    # use get_adj to preprocess data: we should do this for each graph saparately
    # =============================================================================  
    
    # adj_type = 'appr'
    if adj_type == 'un':    
        print("\n Processing to undirected adj")
        indices = edge_index
        features = x
        indices = to_undirected(indices)
        
        edge_index, edge_attr = get_undirected_adj(edge_index = indices,
                                                   num_nodes = features.shape[0],
                                                   dtype = features.dtype)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
    elif adj_type == 'appr':
        print("\n Processing approximate personalized pagerank adj matrix")
        alpha = 0.1
        indices = edge_index
        features = x
        
        edge_index, edge_attr = get_appr_directed_adj(alpha = alpha, 
                                                      edge_index = indices, 
                                                      num_nodes = features.shape[0],
                                                      dtype = features.dtype,
                                                      edge_weight = edge_attr)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
         
    elif adj_type == 'ib':
        print("\n Processing first and second-order adj matrix")
        alpha = 0.1
        indices = edge_index
        features = x
                
        # edge_index, edge_attr = get_appr_directed_adj_keep_attr(alpha = alpha, 
        #                                                         edge_index = indices, 
        #                                                         num_nodes = features.shape[0],
        #                                                         dtype = features.dtype,
        #                                                         edge_weight = edge_attr) 
        
 
        edge_index, edge_attr = get_appr_directed_adj(alpha = alpha, 
                                                      edge_index = indices, 
                                                      num_nodes = features.shape[0],
                                                      dtype = features.dtype,
                                                      edge_weight = edge_attr)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        ##we should input approximate edge_index, edge_attr or the original edge_index, edge_attr?
        
        # edge_index2, edge_attr2 = get_second_directed_adj(edge_index = indices, 
        #                                                   num_nodes = features.shape[0],
        #                                                   dtype = features.dtype,
        #                                                   edge_weight = edge_attr)
        
        edge_index2, edge_attr2 = get_second_directed_adj(edge_index = edge_index, 
                                                          num_nodes = features.shape[0],
                                                          dtype = features.dtype,
                                                          edge_weight = edge_attr)
    
        data.edge_index2 = edge_index2
        data.edge_attr2 = edge_attr2

    return data, is_empty_index


def process_graph(dataset_name, adj_type='ib'):
    """Process the graph using the specified adjacency type"""
    import torch
    
    temp_dir = get_dataset_path(dataset_name, subdir="Graph/TempRaw")
    
    if adj_type not in ['un', 'appr', 'ib']:
        print(f"Warning: Unknown adjacency type '{adj_type}'. Falling back to 'ib'.")
        adj_type = 'ib'
    
    print(f"\nProcessing graph with adjacency type: {adj_type}")
    
    edge_index = read_file(temp_dir, dataset_name, 'A', torch.long).t() - 1
    
    if len(edge_index.shape) == 1 or edge_index.size(1) == 0:
        print("Empty graph detected, skipping...")
        return None, True
        
    batch = read_file(temp_dir, dataset_name, 'graph_indicator', torch.long) - 1
    node_attributes = read_file(temp_dir, dataset_name, 'node_attributes')
    edge_attributes = read_file(temp_dir, dataset_name, 'edge_attributes')
    y = read_file(temp_dir, dataset_name, 'graph_labels', torch.long)
    
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)
    
    x = node_attributes
    edge_attr = edge_attributes
    
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    
    alpha = 0.1  
    
    if adj_type == 'un':
        print("Using undirected graph processing")
        indices = to_undirected(edge_index)
        edge_index, edge_attr = get_undirected_adj(
            edge_index=indices,
            num_nodes=x.shape[0],
            dtype=x.dtype
        )
        
    elif adj_type == 'appr':
        print("Using approximate personalized PageRank processing")
        edge_index, edge_attr = get_appr_directed_adj(
            alpha=alpha,
            edge_index=edge_index,
            num_nodes=x.shape[0],
            dtype=x.dtype,
            edge_weight=edge_attr
        )
        
    elif adj_type == 'ib':
        print("Using first and second-order adjacency processing")
        edge_index, edge_attr = get_appr_directed_adj(
            alpha=alpha,
            edge_index=edge_index,
            num_nodes=x.shape[0],
            dtype=x.dtype,
            edge_weight=edge_attr
        )
        
        edge_index2, edge_attr2 = get_second_directed_adj(
            edge_index=edge_index,
            num_nodes=x.shape[0],
            dtype=x.dtype,
            edge_weight=edge_attr
        )
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.edge_index2 = edge_index2
        data.edge_attr2 = edge_attr2
        
        return data, False
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data, False


def generate_graphs(dataset, num_samples=10000, anomaly_percentage=0.05, adj_type='ib'):
    """Generate graphs for a dataset
    
    Parameters:
    -----------
    dataset : str
        Dataset name
    num_samples : int
        Number of samples to generate
    anomaly_percentage : float
        Percentage of anomalies in the samples (0.0-1.0)
    adj_type : str
        Adjacency type to use ('un', 'appr', or 'ib')
    """
    # Create necessary directories
    raw_dir = get_dataset_path(dataset, subdir="Graph/Raw")
    temp_dir = get_dataset_path(dataset, subdir="Graph/TempRaw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load dataset
    log_df, embedding_df = load_dataset(dataset)
    
    # Sample groups
    sampled_groups, sampled_indices = sample_groups(dataset, num_samples, anomaly_percentage)
    
    # Track events to ensure consistent node indexing
    all_events = []
    graph_count = 0
    
    # Process each group
    for idx, group_name in enumerate(tqdm(sampled_groups, desc=f"Generating {dataset} graphs")):
        # Construct graph for this group
        construct_graph(
            log_df=log_df,
            group_name=group_name,
            embedding_df=embedding_df,
            dataset_name=dataset,
            graph_index=graph_count,
            graph_location_index=sampled_indices[idx]
        )
        
        # Process the graph (apply adjacency transformations)
        graph_data, is_empty = process_graph(dataset, adj_type=adj_type)
        
        if not is_empty:
            # Concatenate with previous graphs
            concatenate_graphs(graph_data, graph_count, dataset, adj_type)
            
            # Update events list and graph count
            new_events = list(dict.fromkeys(log_df[log_df["GroupId"] == group_name]["EventTemplate"]))
            all_events.extend(new_events)
            all_events = list(dict.fromkeys(all_events))
            graph_count += 1
    
    print(f"Generated {graph_count} graphs for {dataset} using {adj_type} adjacency type")
    
    # Copy files to Raw directory for DataLoader
    dest_dir = get_dataset_path(dataset, subdir="Raw")
    os.makedirs(dest_dir, exist_ok=True)
    
    for filename in os.listdir(raw_dir):
        shutil.copy(
            os.path.join(raw_dir, filename),
            os.path.join(dest_dir, filename)
        )

def concatenate_graphs(graph_data, graph_index, dataset_name, adj_type='ib'):
    """Concatenate graphs by appending to existing files"""
    raw_dir = get_dataset_path(dataset_name, subdir="Graph/Raw")
    
    # Write edge indices
    edge_indices = pd.DataFrame(graph_data.edge_index.numpy()).T
    with open(os.path.join(raw_dir, f"{dataset_name}_A.txt"), "ab") as f:
        np.savetxt(f, edge_indices.values, fmt='%i', delimiter=', ')
    
    # Write edge attributes
    edge_attrs = pd.DataFrame(graph_data.edge_attr.numpy())
    with open(os.path.join(raw_dir, f"{dataset_name}_edge_attributes.txt"), "ab") as f:
        np.savetxt(f, edge_attrs.values, fmt='%f', delimiter=', ')

    if adj_type == 'ib' and hasattr(graph_data, 'edge_index2') and graph_data.edge_index2 is not None:
        edge_indices2 = pd.DataFrame(graph_data.edge_index2.numpy()).T
        with open(os.path.join(raw_dir, f"{dataset_name}_A2.txt"), "ab") as f:
            np.savetxt(f, edge_indices2.values, fmt='%i', delimiter=', ')
        
        edge_attrs2 = pd.DataFrame(graph_data.edge_attr2.numpy())
        with open(os.path.join(raw_dir, f"{dataset_name}_edge_attributes2.txt"), "ab") as f:
            np.savetxt(f, edge_attrs2.values, fmt='%f', delimiter=', ')
            
    # Write graph indicator
    num_nodes = graph_data.x.shape[0]
    graph_indicator = pd.DataFrame({"indicator": [graph_index + 1] * num_nodes})
    with open(os.path.join(raw_dir, f"{dataset_name}_graph_indicator.txt"), "ab") as f:
        np.savetxt(f, graph_indicator.values, fmt='%i', delimiter=', ')
    
    # Write graph labels
    graph_labels = pd.DataFrame({"labels": [graph_data.y.item()]})
    with open(os.path.join(raw_dir, f"{dataset_name}_graph_labels.txt"), "ab") as f:
        np.savetxt(f, graph_labels.values, fmt='%i', delimiter=', ')
    
    # Write node attributes
    node_attrs = pd.DataFrame(graph_data.x.numpy())
    with open(os.path.join(raw_dir, f"{dataset_name}_node_attributes.txt"), "ab") as f:
        np.savetxt(f, node_attrs.values, fmt='%f', delimiter=', ')
    
    # Write second-order information if available (primarily for 'ib' adjacency type)
    if adj_type == 'ib' and hasattr(graph_data, 'edge_index2') and graph_data.edge_index2 is not None:
        edge_indices2 = pd.DataFrame(graph_data.edge_index2.numpy()).T
        with open(os.path.join(raw_dir, f"{dataset_name}_A2.txt"), "ab") as f:
            np.savetxt(f, edge_indices2.values, fmt='%i', delimiter=', ')
        
        edge_attrs2 = pd.DataFrame(graph_data.edge_attr2.numpy())
        with open(os.path.join(raw_dir, f"{dataset_name}_edge_attributes2.txt"), "ab") as f:
            np.savetxt(f, edge_attrs2.values, fmt='%f', delimiter=', ')
def ensure_directories(dataset_name):
    """Ensure all required directories exist"""
    dirs = [
        get_dataset_path(dataset_name, subdir="Graph"),
        get_dataset_path(dataset_name, subdir="Graph/Raw"),
        get_dataset_path(dataset_name, subdir="Graph/TempRaw"),
        get_dataset_path(dataset_name, subdir="Raw")
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def main():
    """Main function to execute graph generation pipeline"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate graphs from log data')
    parser.add_argument('--dataset', type=str, default='Kubelet', 
                        help='Dataset name (e.g., Kubelet, HDFS)')
    parser.add_argument('--samples', type=int, default=1000, 
                        help='Number of graph samples to generate')
    parser.add_argument('--anomaly_pct', type=float, default=0.05, 
                        help='Percentage of anomalies in the samples (0.0-1.0)')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Root path for Logs2Graph data (overrides environment variable)')
    parser.add_argument('--adj_type', type=str, default='ib', choices=['un', 'appr', 'ib'],
                        help='Adjacency type: un (undirected), appr (approx. PageRank), ib (first+second order)')
    
    args = parser.parse_args()
    
    # Set root path if provided
    set_root_path(args.root_path)
    
    # Ensure required directories exist
    ensure_directories(args.dataset)
    
    # Print configuration
    print(f"\nGenerating graphs for {args.dataset} dataset")
    print(f"Root path: {get_root_path()}")
    print(f"Number of samples: {args.samples}")
    print(f"Anomaly percentage: {args.anomaly_pct*100:.1f}%")
    print(f"Adjacency type: {args.adj_type}")
    
    # Generate graphs
    generate_graphs(
        dataset=args.dataset,
        num_samples=args.samples,
        anomaly_percentage=args.anomaly_pct,
        adj_type=args.adj_type
    )
    
    print("Graph generation completed successfully!")
# This block will execute when the script is run directly
if __name__ == "__main__":
    main()
