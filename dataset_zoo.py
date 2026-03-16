import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import networkx as nx

import torch_scatter
from torch_geometric.utils import degree, get_ppr
from community import community_louvain
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 24027277

def get_pagerank_features(edge_index, num_nodes, alpha=0.15):
    ppr_edge_index, ppr_weights = get_ppr(edge_index=edge_index, alpha=alpha, num_nodes=num_nodes)
    return torch_scatter.scatter_add(ppr_weights, ppr_edge_index[1], dim=0, dim_size=num_nodes).reshape(-1, 1)


def get_degree_features(edge_index, num_nodes):
    # Calculate in/out degree
    out_deg = degree(edge_index[0], num_nodes)
    in_deg = degree(edge_index[1], num_nodes)
    
    # Total degree and ratio
    total_deg = in_deg + out_deg
    in_out_ratio = in_deg / (out_deg + 1e-8)
    
    # Log normalization (handle power law)
    in_deg_log = torch.log1p(in_deg)
    out_deg_log = torch.log1p(out_deg)
    total_deg_log = torch.log1p(total_deg)
    
    # Ranking feature
    total_deg_rank = torch.argsort(torch.argsort(total_deg, descending=True)).float() / num_nodes

    return torch.stack([
        in_deg, out_deg, in_deg_log, out_deg_log, total_deg_log, in_out_ratio, total_deg_rank
    ], dim=1)


def get_louvain_features(edge_index, num_nodes, labels=None, train_mask=None, resolution=1.0):
    # Build graph using NetworkX (undirected)
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
    
    # Calculate internal degree (edges within same community)
    row, col = edge_index
    same_comm_mask = (torch.from_numpy(comm_ids[row]) == torch.from_numpy(comm_ids[col]))
    internal_edge_index = edge_index[:, same_comm_mask]
    internal_deg = degree(internal_edge_index[0], num_nodes)
    total_deg = degree(edge_index[0], num_nodes)
    
    # Combine features
    louvain_feat = torch.zeros((num_nodes, 5))
    
    for i in range(num_nodes):
        cid = comm_ids[i]
        if cid == -1:
            continue
        
        size = comm_size.get(cid, 1)
        illicit_cnt = comm_train_illicit.get(cid, 0)
        train_total = comm_train_total.get(cid, 1e-8)
        
        louvain_feat[i, 0] = np.log1p(size)                    # Community size (log)
        louvain_feat[i, 1] = illicit_cnt / train_total          # Illicit ratio
        louvain_feat[i, 2] = 1.0 if illicit_cnt > 0 else 0.0   # Has illicit flag
        louvain_feat[i, 3] = internal_deg[i] / (total_deg[i] + 1e-8)  # Internal degree ratio
        louvain_feat[i, 4] = internal_deg[i] / (size)          # Average internal degree
    
    return louvain_feat, partition


def get_standard_scaler(data):
    scaler = StandardScaler()
    x_numpy = data.x
    scaler.fit(x_numpy[data.train_mask]) # 記住咗 Train Set 的 Mean 同 Std
    return scaler.transform(x_numpy)  # Fit and transform
    


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
    # print(y)
    # print(x)
    # print(f"Feature shape: {x.shape}, Label shape: {y.shape}")
    # print(nodes_df.head())

    tx_id_map = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'])}
    edge_index_src = edgelist_df.iloc[:, 0].map(tx_id_map)
    edge_index_tgt = edgelist_df.iloc[:, 1].map(tx_id_map)
    edges = pd.concat([edge_index_src, edge_index_tgt], axis=1).astype(int)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    # print('edge_index_src', edge_index_src.shape)
    # print('edge_index_tgt', edge_index_tgt.shape)
    
    # print('edge_index_src nulls:', edge_index_src.isnull().sum().sum())
    # print('edge_index_tgt nulls:', edge_index_tgt.isnull().sum().sum())
    # print('edges nulls:', edges.isnull().sum().sum())

    # print('\nedges:', edges)
    # print(f"Edge index shape:\n {edge_index.shape}")
    # print(f"Edge index:\n {edge_index}")

    # 6. New Data and create Mask -------------------------------
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)
    data.train_mask = (data.timesteps < 35) & (y != -1)
    data.val_mask   = (data.timesteps >= 35) & (data.timesteps < 42) & (y != -1)
    data.test_mask  = (data.timesteps >= 42) & (y != -1)
    # print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

    # 7. feature engineering - pagerank, degree, louvain -------------------------------
    print("PageRank features...")
    pagerank_features = get_pagerank_features(data.edge_index, data.x.size(0))

    print("Degree features...")
    degree_features = get_degree_features(data.edge_index, data.x.size(0))

    print("Louvain features...")
    louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)

    print("combine features to data.x...")
    data.x = torch.cat([data.x, pagerank_features, degree_features, louvain_features], dim=1)
    print(f"Total features: {data.x.size(1)} dimensions")

    # 7. StandardScaler ----------------------------------------------------
    print("\nStandardScaler...")
    x_scaled = get_standard_scaler(data)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    print(f"StandardScaler done...\nTotal features: {data.x.size(1)} dimensions")

    # 8. Save processed data
    torch.save(data, dataset_dir+'/elliptic_processed_data.pt')

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

def load_elliptic_data(fname='Dataset/elliptic dataset/elliptic_processed_data.pt'):
    print("\n1. Loading Elliptic dataset...")

    data = torch.load(fname, weights_only=False)
    print(data.train_mask.sum()) 
    print(data.test_mask.sum())
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
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

    # 2. process nodes
    phishing_addresses, non_phishing_addresses = [], []
    node_mapping = {}
    y_list = []
    x_np = np.zeros((G.number_of_nodes(), 12), dtype=np.float32)
    for i, (n, attr) in enumerate(G.nodes(data=True)):
        # node_mapping for edge_index construction later
        node_mapping[n] = i

        # X extract the nodes and edges for the graph
        # out-degree, in-degree, average degree, total degree, 
        # average sending amount, total sending amount, maximumsending amount, 
        # average receiving amount, total receiving amount, maximum receiving amount, 
        # transaction time interval ratio, and the total number of neighbors
        # --- Degree ---
        in_d = G.in_degree(n)
        out_d = G.out_degree(n)
        t_send, m_send, t_recv, m_recv = 0, 0, 0, 0
        all_ts = []
        
        for _, _, d in G.in_edges(n, data=True):
            amt = d.get('amount', 0)
            t_recv += amt
            if amt > m_recv: m_recv = amt
            all_ts.append(d.get('timestamp', 0))
            
        for _, _, d in G.out_edges(n, data=True):
            amt = d.get('amount', 0)
            t_send += amt
            if amt > m_send: m_send = amt
            all_ts.append(d.get('timestamp', 0))

        # Time Ratio
        if len(all_ts) > 1:
            all_ts.sort()
            t_ratio = (all_ts[-1] - all_ts[0]) / len(all_ts)
        else:
            t_ratio = 0
            
        x_np[i] = [
            in_d, out_d, (in_d+out_d)/2, in_d+out_d,
            t_send/out_d if out_d > 0 else 0, t_send, m_send,
            t_recv/in_d if in_d > 0 else 0, t_recv, m_recv,
            t_ratio, len(list(G.neighbors(n)))
        ]

        # y extract the labels for the nodes
        y_list.append(attr.get("isp", 0))  # 默認為 0 (non-phishing) 如果沒有 isp 屬性

        # count phishing vs non-phishing for EDA for plt
        if attr.get("isp") == 1:
            phishing_addresses.append(n)
        else:
            non_phishing_addresses.append(n)

    y = torch.tensor(y_list, dtype=torch.long)
    x = torch.from_numpy(x_np)
    x = torch.log(x + 1)


    # 3. process edges
    num_edges = G.number_of_edges()
    edges_np = np.zeros((num_edges, 2), dtype=np.int64)
    edge_attr_np = np.zeros((num_edges, 2), dtype=np.float32)
    for i, (u, v, attr) in enumerate(G.edges(data=True)):
        edges_np[i] = [node_mapping[u], node_mapping[v]]
        edge_attr_np[i] = [attr.get('amount', 0), attr.get('timestamp', 0)]
    edge_index = torch.from_numpy(edges_np).t().contiguous()
    edge_attr = torch.from_numpy(edge_attr_np)

    # 4. plt ----------------------------------------------------
    # plt.figure(figsize=(8, 6))
    # sns.barplot(x=[0, 1], y=[len(non_phishing_addresses), len(phishing_addresses)])
    # plt.xticks([0, 1], ["Non-Phishing", "Phishing"])
    # plt.yscale('log')
    # plt.title("Class Distribution")
    # plt.ylabel("Number of Addresses")
    # plt.xlabel("Category")
    # for i, v in enumerate([len(non_phishing_addresses), len(phishing_addresses)]):
    #     plt.text(i, v, f'{v}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    # plt.show()

    # 5. New Data ----------------------------------------------------
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(data.num_nodes * 0.6)] = True
    data.val_mask[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)] = True
    data.test_mask[int(data.num_nodes * 0.8):] = True
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    

    # 6. Louvain ----------------------------------------------------
    print("Louvain features...")
    louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)
    data.x = torch.cat([data.x, louvain_features], dim=1)
    print(f"Total features: {data.x.size(1)} dimensions")

    # 7. StandardScaler ----------------------------------------------------
    print("\nStandardScaler...")
    x_scaled = get_standard_scaler(data)
    data.x = torch.tensor(x_scaled, dtype=torch.float)
    print(f"StandardScaler done...\nTotal features: {data.x.size(1)} dimensions")

    # 8. Save processed data
    torch.save(data, dataset_dir + '/ethereum_processed_data.pt')


def load_ethereum_data(fname='Dataset/ethereum dataset/ethereum_processed_data.pt'):
    print("\n1. Loading Ethereum dataset...")

    data = torch.load(fname, weights_only=False)
    print(data.train_mask.sum()) 
    print(data.test_mask.sum())
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
    return data

# process_ethereum_data()
process_elliptic_data()