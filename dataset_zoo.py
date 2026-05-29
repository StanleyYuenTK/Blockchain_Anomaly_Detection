"""
The Hong Kong Polytechnic University
Student ID: 24027277d
Name: Yuen Tsz Ki

Used to process dataset.
"""

import os
import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

import networkx as nx
import torch
from torch_geometric.data import Data
import torch_scatter
from torch_geometric.utils import degree, get_ppr
from community import community_louvain
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 24027277


# ========================
# Personal page rank
# ========================
def get_pagerank_features(edge_index, num_nodes, alpha=0.15):
    ppr_edge_index, ppr_weights = get_ppr(edge_index=edge_index, alpha=alpha, num_nodes=num_nodes)
    return torch_scatter.scatter_add(ppr_weights, ppr_edge_index[1], dim=0, dim_size=num_nodes).reshape(-1, 1)


# ========================
# Degree
# ========================
def get_degree_features(edge_index, num_nodes):
    # Calculate in/out degree
    out_deg = degree(edge_index[0], num_nodes)
    in_deg = degree(edge_index[1], num_nodes)
    total_deg = in_deg + out_deg
    ratio_deg = torch.where(out_deg > 0, in_deg / out_deg, torch.zeros_like(in_deg)) # prevent node with no outgoing edge
    
    # Log normalization
    in_deg_log = torch.log1p(in_deg)
    out_deg_log = torch.log1p(out_deg)
    total_deg_log = torch.log1p(total_deg)
    ratio_deg_log = torch.log1p(ratio_deg)

    return torch.stack([in_deg_log, out_deg_log, total_deg_log, ratio_deg_log], dim=1)


# ========================
# Louvain Features
# ========================
def get_louvain_features(edge_index, num_nodes, labels=None, train_mask=None, resolution=1.0):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().numpy())
    
    # Run Louvain community detection
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=RANDOM_SEED)
    
    # Get community IDs for all nodes
    comm_ids = np.array([partition.get(i, -1) for i in range(num_nodes)])
    
    # Compute community statistics
    comm_size = {}
    comm_train_illicit = {}
    comm_train_total = {}
    
    labels_np = labels.numpy() if labels is not None else np.zeros(num_nodes)
    train_mask_np = train_mask.numpy() if train_mask is not None else np.ones(num_nodes, dtype=bool)
    
    # Calculate community stats (only on train set to prevent leakage)
    for i in range(num_nodes):
        cid = comm_ids[i]
        if cid == -1:
            continue
        
        comm_size[cid] = comm_size.get(cid, 0) + 1
        
        if train_mask_np[i]:
            if labels_np[i] == 1:  # Illicit
                comm_train_illicit[cid] = comm_train_illicit.get(cid, 0) + 1
            comm_train_total[cid] = comm_train_total.get(cid, 0) + 1
    
    # Combine features
    louvain_feat = torch.zeros((num_nodes, 3))
    
    for i in range(num_nodes):
        cid = comm_ids[i]
        if cid == -1:
            continue
        
        size = comm_size.get(cid, 1)
        illicit_cnt = comm_train_illicit.get(cid, 0)
        train_total = comm_train_total.get(cid, 1e-8)
        
        louvain_feat[i, 0] = np.log1p(size)                     # Community size (log)
        louvain_feat[i, 1] = illicit_cnt / train_total          # Illicit ratio
        louvain_feat[i, 2] = 1.0 if illicit_cnt > 0 else 0.0    # Has illicit flag
   
    return louvain_feat, partition


# ========================
# community features
# ========================
def get_clustering_features(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().cpu().numpy())

    cluster_dict = nx.clustering(G)
    clustering = torch.tensor([cluster_dict.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float).view(-1, 1)
    clustering = torch.log1p(clustering)
    return clustering 


# ========================
# scaler
# ========================
def get_standard_scaler(data):
    scaler = StandardScaler()
    x_numpy = data.x
    scaler.fit(x_numpy[data.train_mask])
    return scaler.transform(x_numpy)


# =========================================================================================
# Elliptic dataset
# =========================================================================================
def process_elliptic_data(dataset_dir='Dataset/elliptic dataset'):
    print("\n1. Processing Elliptic dataset...")

    # 1. load data -------------------------------
    classes_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_classes.csv'))
    edgelist_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv'))
    features_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_features.csv'), header=None)

    # 2. colA is id, colB is time steps -------------------------------
    # features without hearder, so we assign column names first for merge
    features_df.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(2, features_df.shape[1])]
    
    # 3. merge features and classes to become a nodes -------------------------------
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    
    # 4. process features and labels and edges for push into PyG data object -------------------------------
    # class 2 = licit, class 1 = illicit, unknown -> -1
    y = torch.tensor(nodes_df['class'].map({'2': 0, '1': 1}).fillna(-1).values, dtype=torch.long)

    # get x without txId, timestep and target(class), 
    # since my model not need time to learn, so we drop them
    x = torch.tensor(nodes_df.iloc[:, 2:-1].values, dtype=torch.float)

    tx_id_map = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'])}
    edge_index_src = edgelist_df.iloc[:, 0].map(tx_id_map)
    edge_index_tgt = edgelist_df.iloc[:, 1].map(tx_id_map)
    edges = pd.concat([edge_index_src, edge_index_tgt], axis=1).astype(int)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # 6. New Data and create Mask -------------------------------
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)
    data.train_mask = (data.timesteps < 35) & (y != -1)
    data.val_mask   = (data.timesteps >= 35) & (data.timesteps < 42) & (y != -1)
    data.test_mask  = (data.timesteps >= 42) & (y != -1)

    # 7. feature engineering - pagerank, degree, louvain -------------------------------
    print("PageRank features...")
    pagerank_features = get_pagerank_features(data.edge_index, data.x.size(0))

    print("Degree features...")
    degree_features = get_degree_features(data.edge_index, data.x.size(0))

    print("Louvain features...")
    louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)
    data.comm_id = torch.tensor([partition.get(i, 0) for i in range(data.x.size(0))], dtype=torch.long)
    print("data.comm_id","="*60, "\n", data.comm_id)

    print("Community features")
    clustering_features = get_clustering_features(data.edge_index, data.x.size(0))

    print("combine features to data.x...")
    data.x = torch.cat([data.x, pagerank_features, degree_features, louvain_features, clustering_features], dim=1)
    print(f"Total features: {data.x.size(1)} dimensions")

    # 7. StandardScaler ----------------------------------------------------
    print("\nStandardScaler...")
    x_scaled = get_standard_scaler(data)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    print(f"StandardScaler done...\nTotal features: {data.x.size(1)} dimensions")

    # 8. Save processed data
    torch.save(data, dataset_dir+'/elliptic_processed_data.pt')


def process_elliptic_data_noL(dataset_dir='Dataset/elliptic dataset'):
    print("\n1. Processing Elliptic dataset...")

    # 1. load data -------------------------------
    classes_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_classes.csv'))
    edgelist_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv'))
    features_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_features.csv'), header=None)

    # 2. colA is id, colB is time steps -------------------------------
    # features without hearder, so we assign column names first for merge
    features_df.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(2, features_df.shape[1])]
    
    # 3. merge features and classes to become a nodes -------------------------------
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    
    # 4. process features and labels and edges for push into PyG data object -------------------------------
    # class 2 = licit, class 1 = illicit, unknown -> -1
    y = torch.tensor(nodes_df['class'].map({'2': 0, '1': 1}).fillna(-1).values, dtype=torch.long)

    # get x without txId, timestep and target(class), 
    # since my model not need time to learn, so we drop them
    x = torch.tensor(nodes_df.iloc[:, 2:-1].values, dtype=torch.float)

    tx_id_map = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'])}
    edge_index_src = edgelist_df.iloc[:, 0].map(tx_id_map)
    edge_index_tgt = edgelist_df.iloc[:, 1].map(tx_id_map)
    edges = pd.concat([edge_index_src, edge_index_tgt], axis=1).astype(int)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # 6. New Data and create Mask -------------------------------
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)
    data.train_mask = (data.timesteps < 35) & (y != -1)
    data.val_mask   = (data.timesteps >= 35) & (data.timesteps < 42) & (y != -1)
    data.test_mask  = (data.timesteps >= 42) & (y != -1)

    # 7. feature engineering - pagerank, degree, louvain -------------------------------
    print("PageRank features...")
    pagerank_features = get_pagerank_features(data.edge_index, data.x.size(0))

    print("Degree features...")
    degree_features = get_degree_features(data.edge_index, data.x.size(0))

    # print("Louvain features...")
    # louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)

    print("combine features to data.x...")
    data.x = torch.cat([data.x, pagerank_features, degree_features], dim=1)
    print(f"Total features: {data.x.size(1)} dimensions")

    # 7. StandardScaler ----------------------------------------------------
    print("\nStandardScaler...")
    x_scaled = get_standard_scaler(data)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    print(f"StandardScaler done...\nTotal features: {data.x.size(1)} dimensions")

    # 8. Save processed data
    torch.save(data, dataset_dir+'/elliptic_processed_data_noL.pt')


def eda_elliptic(dataset_dir='Dataset/elliptic dataset'):
    # 1. load data
    classes_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_classes.csv'))
    edgelist_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv'))
    features_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_features.csv'), header=None)
    
    # Perform exploratory data analysis on the loaded data
    print("Elliptic Data Analysis:")
    print(f"Number of nodes: {features_df.shape[0]}")
    print(f"Number of edges: {edgelist_df.shape[0]}")
    print(f"Number of features: {features_df.shape[1] - 2}")  # Exclude txId and timestep

    # check missing values
    print(f"Missing values in features: {features_df.isnull().sum().sum()}")
    print(f"Missing values in classes: {classes_df.isnull().sum().sum()}")
    print(f"Missing values in edgelist: {edgelist_df.isnull().sum().sum()}")
    print("Features null:", features_df.isnull().values.any())
    print("Classes null:", classes_df.isnull().values.any())
    print("Edges null:", edgelist_df.isnull().values.any())

    # 2. colA is id, colB is time steps
    # features without hearder, so we assign column names first for merge
    features_df.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(2, features_df.shape[1])]

    # merge features and classes to become a nodes
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    nodes_df['label_class'] = nodes_df['class'].replace({
        'unknown':-1, 
        '2': 0,  # licit
        '1': 1,  # illicit
    })
    print(nodes_df.head())

    # 3. class distribution
    class_counts = nodes_df['label_class'].value_counts()
    print(class_counts)
    plt.figure(figsize=(6,4))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks([0,1,2], ["Unknown","Licit","Illicit"])
    plt.title("Class Distribution")
    plt.ylabel("Number of Transactions")
    plt.xlabel("Class")
    plt.show()

    y = torch.tensor(nodes_df['label_class'].values, dtype=torch.long)
    print(y)


def load_elliptic_data(fname='Dataset/elliptic dataset/elliptic_processed_data.pt', ver=''):
    if ver == 'nol':
        fname = 'Dataset/elliptic dataset/elliptic_processed_data_noL.pt'

    print(f"\n1. Loading Elliptic dataset {ver}: {fname}...")

    data = torch.load(fname, weights_only=False)
    y = data.y

    masks = {
        'train': data.train_mask,
        'val': data.val_mask,
        'test': data.test_mask
    }
    for name, mask in masks.items():
        y_masked = y[mask]
        total = len(y_masked)
        licit = (y_masked == 0).sum().item()
        illicit = (y_masked == 1).sum().item()
        illicit_ratio = illicit / (total if total > 0 else 0)
        print(f"Mask {name}:, Train set: Total={total}, Licit={licit}, Illicit={illicit}, Illicit Ratio={illicit_ratio}")

    return data
    

# =========================================================================================
# Ethereum dataset
# =========================================================================================
def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def process_ethereum_data(dataset_dir='Dataset/ethereum dataset'):
    print("\n1. Processing Ethereum dataset...")
    # 1. load data
    G = load_pickle(dataset_dir+'/MulDiGraph.pkl')

    # 2. get phishing nodes for subgraph sampling
    #    Licit nodes are also included to ensure fairness, 
    #    all nodes in the subgraph will be related to phishing nodes, 
    phishing_nodes, licit_nodes = [], [] 
    for n, attr in G.nodes(data=True):
        if attr.get('isp') == 1:
            phishing_nodes.append(n)
        else:
            licit_nodes.append(n)
    
    rng = np.random.default_rng(24027277)
    rng_licit_nodes = rng.choice(licit_nodes, size=len(phishing_nodes), replace=False).tolist()

    all_nodes = set(phishing_nodes) | set(rng_licit_nodes)
    subgraph_nodes = set(all_nodes)
    for node in all_nodes:
        subgraph_nodes.update(G.neighbors(node))

    S = G.subgraph(subgraph_nodes).copy()

    # 2. process nodes
    node_mapping = {}
    y_list = []
    x_np = np.zeros((S.number_of_nodes(), 12), dtype=np.float32)
    for i, (n, attr) in enumerate(S.nodes(data=True)):
        # node_mapping for edge_index construction later
        node_mapping[n] = i

        # X extract the nodes and edges for the graph
        # out-degree, in-degree, average degree, total degree, 
        # average sending amount, total sending amount, maximum sending amount, 
        # average receiving amount, total receiving amount, maximum receiving amount, 
        # transaction time interval ratio, and the total number of neighbors
        # --- Degree ---
        in_deg = S.in_degree(n)
        out_deg = S.out_degree(n)
        total_send, max_send, total_recv, max_recv = 0, 0, 0, 0
        all_ts = []
        
        for _, _, d in S.in_edges(n, data=True):
            amt = d.get('amount', 0)
            total_recv += amt
            if amt > max_recv: max_recv = amt
            all_ts.append(d.get('timestamp', 0))
            
        for _, _, d in S.out_edges(n, data=True):
            amt = d.get('amount', 0)
            total_send += amt
            if amt > max_send: max_send = amt
            all_ts.append(d.get('timestamp', 0))

        # Time Ratio
        if len(all_ts) > 1:
            all_ts.sort()
            time_ratio = (all_ts[-1] - all_ts[0]) / len(all_ts)
        else:
            time_ratio = 0
            
        x_np[i] = [
            in_deg, out_deg, (in_deg+out_deg)/2, in_deg+out_deg,
            total_send/out_deg if out_deg > 0 else 0, total_send, max_send,
            total_recv/in_deg if in_deg > 0 else 0, total_recv, max_recv,
            time_ratio, len(list(S.neighbors(n)))
        ]

        # y extract the labels for the nodes
        y_list.append(attr.get("isp", 0))

    y = torch.tensor(y_list, dtype=torch.long)
    x = torch.from_numpy(x_np)
    x = torch.log(x + 1)


    # 3. process edges
    num_edges = S.number_of_edges()
    edges_np = np.zeros((num_edges, 2), dtype=np.int64)
    edge_attr_np = np.zeros((num_edges, 2), dtype=np.float32)
    for i, (u, v, attr) in enumerate(S.edges(data=True)):
        edges_np[i] = [node_mapping[u], node_mapping[v]]
        edge_attr_np[i] = [attr.get('amount', 0), attr.get('timestamp', 0)]
    edge_index = torch.from_numpy(edges_np).t().contiguous()
    edge_attr = torch.from_numpy(edge_attr_np)

    # 4. New Data ----------------------------------------------------
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(data.num_nodes * 0.6)] = True
    data.val_mask[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)] = True
    data.test_mask[int(data.num_nodes * 0.8):] = True
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
    # 5. feature engineering - pagerank, louvain, degree already added to data -------------------------------
    print("PageRank features...")
    pagerank_features = get_pagerank_features(data.edge_index, data.x.size(0))

    # 6. Louvain ----------------------------------------------------
    print("Louvain features...")
    louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)
    data.comm_id = torch.tensor([partition.get(i, 0) for i in range(data.x.size(0))], dtype=torch.long)
    print("data.comm_id","="*60, "\n", data.comm_id)

    print("Community features")
    clustering_features = get_clustering_features(data.edge_index, data.x.size(0))

    # 7. combine
    data.x = torch.cat([data.x, pagerank_features, louvain_features, clustering_features], dim=1)
    print(f"Total features: {data.x.size(1)} dimensions")

    # 7. StandardScaler ----------------------------------------------------
    print("\nStandardScaler...")
    x_scaled = get_standard_scaler(data)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    print(f"StandardScaler done...\nTotal features: {data.x.size(1)} dimensions")

    # 8. Save processed data
    torch.save(data, dataset_dir + '/subgraph_ethereum_processed_data.pt')


def process_ethereum_data_noL(dataset_dir='Dataset/ethereum dataset'):
    print("\n1. Processing Ethereum dataset...")
    # 1. load data
    G = load_pickle(dataset_dir+'/MulDiGraph.pkl')

    # 2. get phishing nodes for subgraph sampling
    #    Licit nodes are also included to ensure fairness, 
    #    all nodes in the subgraph will be related to phishing nodes, 
    phishing_nodes, licit_nodes = [], [] 
    for n, attr in G.nodes(data=True):
        if attr.get('isp') == 1:
            phishing_nodes.append(n)
        else:
            licit_nodes.append(n)
    
    rng = np.random.default_rng(24027277)
    rng_licit_nodes = rng.choice(licit_nodes, size=len(phishing_nodes), replace=False).tolist()

    all_nodes = set(phishing_nodes) | set(rng_licit_nodes)
    subgraph_nodes = set(all_nodes)
    for node in all_nodes:
        subgraph_nodes.update(G.neighbors(node))

    S = G.subgraph(subgraph_nodes).copy()

    # 2. process nodes
    node_mapping = {}
    y_list = []
    x_np = np.zeros((S.number_of_nodes(), 12), dtype=np.float32)
    for i, (n, attr) in enumerate(S.nodes(data=True)):
        # node_mapping for edge_index construction later
        node_mapping[n] = i

        # X extract the nodes and edges for the graph
        # out-degree, in-degree, average degree, total degree, 
        # average sending amount, total sending amount, maximum sending amount, 
        # average receiving amount, total receiving amount, maximum receiving amount, 
        # transaction time interval ratio, and the total number of neighbors
        # --- Degree ---
        in_deg = S.in_degree(n)
        out_deg = S.out_degree(n)
        total_send, max_send, total_recv, max_recv = 0, 0, 0, 0
        all_ts = []
        
        for _, _, d in S.in_edges(n, data=True):
            amt = d.get('amount', 0)
            total_recv += amt
            if amt > max_recv: max_recv = amt
            all_ts.append(d.get('timestamp', 0))
            
        for _, _, d in S.out_edges(n, data=True):
            amt = d.get('amount', 0)
            total_send += amt
            if amt > max_send: max_send = amt
            all_ts.append(d.get('timestamp', 0))

        # Time Ratio
        if len(all_ts) > 1:
            all_ts.sort()
            time_ratio = (all_ts[-1] - all_ts[0]) / len(all_ts)
        else:
            time_ratio = 0
            
        x_np[i] = [
            in_deg, out_deg, (in_deg+out_deg)/2, in_deg+out_deg,
            total_send/out_deg if out_deg > 0 else 0, total_send, max_send,
            total_recv/in_deg if in_deg > 0 else 0, total_recv, max_recv,
            time_ratio, len(list(S.neighbors(n)))
        ]

        # y extract the labels for the nodes
        y_list.append(attr.get("isp", 0))  # 默認為 0 (non-phishing) 如果沒有 isp 屬性

    y = torch.tensor(y_list, dtype=torch.long)
    x = torch.from_numpy(x_np)
    x = torch.log(x + 1)


    # 3. process edges
    num_edges = S.number_of_edges()
    edges_np = np.zeros((num_edges, 2), dtype=np.int64)
    edge_attr_np = np.zeros((num_edges, 2), dtype=np.float32)
    for i, (u, v, attr) in enumerate(S.edges(data=True)):
        edges_np[i] = [node_mapping[u], node_mapping[v]]
        edge_attr_np[i] = [attr.get('amount', 0), attr.get('timestamp', 0)]
    edge_index = torch.from_numpy(edges_np).t().contiguous()
    edge_attr = torch.from_numpy(edge_attr_np)

    # 4. New Data ----------------------------------------------------
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(data.num_nodes * 0.6)] = True
    data.val_mask[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)] = True
    data.test_mask[int(data.num_nodes * 0.8):] = True
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
    # 5. feature engineering - pagerank -------------------------------
    print("PageRank features...")
    pagerank_features = get_pagerank_features(data.edge_index, data.x.size(0))

    # 6. Louvain ----------------------------------------------------
    # print("Louvain features...")
    # louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)
    
    data.x = torch.cat([data.x, pagerank_features], dim=1)
    print(f"Total features: {data.x.size(1)} dimensions")

    # 7. StandardScaler ----------------------------------------------------
    print("\nStandardScaler...")
    x_scaled = get_standard_scaler(data)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    print(f"StandardScaler done...\nTotal features: {data.x.size(1)} dimensions")

    # 8. Save processed data
    torch.save(data, dataset_dir + '/subgraph_ethereum_processed_data_noL.pt')


def eda_ethereum(dataset_dir='Dataset/ethereum dataset'):
    print("\n1. EDA Ethereum dataset...")
    # 1. load data
    G = load_pickle(dataset_dir+'/MulDiGraph.pkl') 

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = (2 * num_edges) / num_nodes
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"average degree: {avg_degree:.4f}")

    degrees = [d for n, d in G.degree()]

    node_data = []
    for n, attr in G.nodes(data=True):
        node_data.append(attr)
    df_nodes = pd.DataFrame(node_data)


    print("\n1. number of pishing...")
    counts = df_nodes['isp'].value_counts()
    probs = df_nodes['isp'].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({'count': counts, '%': probs})
    print(dist_df)


    print("\n2. missing value...")
    missing = df_nodes.isnull().sum()
    print("missing", missing[missing > 0])

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    if 'isp' in df_nodes.columns:
        sns.countplot(x='isp', data=df_nodes, ax=ax[0], palette='magma')
        ax[0].set_title('Phishing (1) vs Non-Phishing (0) Distribution')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Count (Log Scale)')

    degrees = [d for n, d in G.degree()]
    sns.histplot(degrees, bins=50, kde=True, ax=ax[1], color='blue')
    ax[1].set_title('Node Degree Distribution')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Degree (Log Scale)')

    plt.tight_layout()
    plt.show()


def load_ethereum_data(fname='Dataset/ethereum dataset/subgraph_ethereum_processed_data.pt', ver=''):
    if ver == 'nol':
        fname = 'Dataset/ethereum dataset/subgraph_ethereum_processed_data_noL.pt'

    print(f"\n1. Loading Ethereum dataset {ver}: {fname}...")

    data = torch.load(fname, weights_only=False)
    y = data.y

    masks = {
        'train': data.train_mask,
        'val': data.val_mask,
        'test': data.test_mask
    }
    for name, mask in masks.items():
        y_masked = y[mask]
        total = len(y_masked)
        licit = (y_masked == 0).sum().item()
        illicit = (y_masked == 1).sum().item()
        illicit_ratio = illicit / (total if total > 0 else 0)
        print(f"Mask {name}:, Train set: Total={total}, Licit={licit}, Illicit={illicit}, Illicit Ratio={illicit_ratio}")
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: e1 or e2')
    args = parser.parse_args()
    if args.dataset == "e1":
        process_elliptic_data()
    
    elif args.dataset == "e1nol":
        process_elliptic_data_noL()

    if args.dataset == "e2":
        process_ethereum_data()
    elif args.dataset == "e2nol":
        process_ethereum_data_noL()