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

import argparse
from torch_geometric.utils import to_networkx

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


def get_standard_scaler(data):
    scaler = StandardScaler()
    x_numpy = data.x
    scaler.fit(x_numpy[data.train_mask]) # 記住咗 Train Set 的 Mean 同 Std
    return scaler.transform(x_numpy)  # Fit and transform
    


# ========================
# Louvain Features
# ========================
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

    print("louvain_feat:\n", louvain_feat)
    
    return louvain_feat, partition


def get_guilt_by_association_feature(edge_index, num_nodes, labels, train_mask):
    """
    計算反向度數加權的連坐特徵 (Guilt-by-Association)
    """
    row, col = edge_index
    # 取得所有節點的度數
    deg = degree(col, num_nodes, dtype=torch.float)
    
    # 計算度數懲罰權重: 1 / log(1 + degree)
    # 度數越大的節點(如交易所)，傳遞的嫌疑值越低
    penalty_weight = 1.0 / torch.log1p(deg)
    
    # 準備 Train set 中的惡意標籤 (僅使用 train_mask 防止洩漏)
    is_illicit = torch.zeros(num_nodes, dtype=torch.float)
    train_mask_bool = train_mask.bool()
    
    # 假設 label 1 是 illicit
    illicit_mask = (labels == 1) & train_mask_bool
    is_illicit[illicit_mask] = 1.0
    
    # 將惡意標籤乘上該節點的懲罰權重
    illicit_msg = is_illicit * penalty_weight
    
    # 透過 edge_index 將訊息傳遞給鄰居 (Message Passing)
    # 這裡計算的是 In-Degree 方向的惡意影響
    guilt_feature = torch_scatter.scatter_add(illicit_msg[row], col, dim=0, dim_size=num_nodes)
    
    return guilt_feature.view(-1, 1)



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

    print("illict features...")
    

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
    # print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

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

    print(f"\n1. Loading Elliptic dataset {ver}...")

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

    # 2. get phishing nodes for subgraph sampling
    phishing_nodes = [n for n, attr in G.nodes(data=True) if attr.get('isp') == 1]

    # get 1-hop neighbours of phishing nodes
    subgraph_nodes = set(phishing_nodes)
    for node in phishing_nodes:
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
        # average sending amount, total sending amount, maximumsending amount, 
        # average receiving amount, total receiving amount, maximum receiving amount, 
        # transaction time interval ratio, and the total number of neighbors
        # --- Degree ---
        in_d = S.in_degree(n)
        out_d = S.out_degree(n)
        t_send, m_send, t_recv, m_recv = 0, 0, 0, 0
        all_ts = []
        
        for _, _, d in S.in_edges(n, data=True):
            amt = d.get('amount', 0)
            t_recv += amt
            if amt > m_recv: m_recv = amt
            all_ts.append(d.get('timestamp', 0))
            
        for _, _, d in S.out_edges(n, data=True):
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
            t_ratio, len(list(S.neighbors(n)))
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

    # 5. New Data ----------------------------------------------------
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(data.num_nodes * 0.6)] = True
    data.val_mask[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)] = True
    data.test_mask[int(data.num_nodes * 0.8):] = True
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
    # 7. feature engineering - pagerank, degree, louvain -------------------------------
    print("PageRank features...")
    pagerank_features = get_pagerank_features(data.edge_index, data.x.size(0))

    # 6. Louvain ----------------------------------------------------
    print("Louvain features...")
    louvain_features, partition = get_louvain_features(data.edge_index, data.x.size(0), labels=data.y, train_mask=data.train_mask)
    
    data.x = torch.cat([data.x, louvain_features, pagerank_features], dim=1)
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
    phishing_nodes = [n for n, attr in G.nodes(data=True) if attr.get('isp') == 1]

    # get 1-hop neighbours of phishing nodes
    subgraph_nodes = set(phishing_nodes)
    for node in phishing_nodes:
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
        # average sending amount, total sending amount, maximumsending amount, 
        # average receiving amount, total receiving amount, maximum receiving amount, 
        # transaction time interval ratio, and the total number of neighbors
        # --- Degree ---
        in_d = S.in_degree(n)
        out_d = S.out_degree(n)
        t_send, m_send, t_recv, m_recv = 0, 0, 0, 0
        all_ts = []
        
        for _, _, d in S.in_edges(n, data=True):
            amt = d.get('amount', 0)
            t_recv += amt
            if amt > m_recv: m_recv = amt
            all_ts.append(d.get('timestamp', 0))
            
        for _, _, d in S.out_edges(n, data=True):
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
            t_ratio, len(list(S.neighbors(n)))
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

    # 5. New Data ----------------------------------------------------
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:int(data.num_nodes * 0.6)] = True
    data.val_mask[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)] = True
    data.test_mask[int(data.num_nodes * 0.8):] = True
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
    # 7. feature engineering - pagerank, degree, louvain -------------------------------
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
    print(f"中位數度數: {np.median(degrees)}")
    print(f"最大度數: {np.max(degrees)}")


    node_data = []
    for n, attr in G.nodes(data=True):
        node_data.append(attr)
    df_nodes = pd.DataFrame(node_data)

    # --- 新增：檢索所有屬性欄位 ---
    print("\n### 0. 欄位屬性 (Attributes/Columns) 檢索 ###")
    
    # 獲取節點屬性範例 (取第一個節點)
    first_node = next(iter(G.nodes()))
    node_keys = G.nodes[first_node].keys()
    print(f"節點屬性 (Node Columns): {list(node_keys)}")
    
    # 獲取邊屬性範例 (取第一條邊)
    if G.number_of_edges() > 0:
        first_edge = next(iter(G.edges(data=True)))
        edge_keys = first_edge[2].keys() # first_edge 為 (u, v, data_dict)
        print(f"邊屬性 (Edge Columns): {list(edge_keys)}")
    # --------------------------------

    print("\n### 2. 節點屬性與標籤分佈 ###")
    if 'isp' in df_nodes.columns:
        counts = df_nodes['isp'].value_counts()
        probs = df_nodes['isp'].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'數量': counts, '百分比 (%)': probs})
        print(dist_df)
    else:
        print("警告：未在節點屬性中找到 'isp' 標籤。")

    # 3. 缺失值檢測 (Missing Data)
    print("\n### 3. 屬性缺失值統計 ###")
    missing = df_nodes.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("所有節點屬性完整，無缺失值。")

    # 4. 邊屬性分析 (EDA on Edges)
    edge_amounts = [d.get('amount', 0) for _, _, d in G.edges(data=True)]
    print("\n### 4. 邊權重 (Amount) 統計 ###")
    print(pd.Series(edge_amounts).describe())

    # 5. 可視化 (Visualization)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # 標籤比例柱狀圖
    if 'isp' in df_nodes.columns:
        sns.countplot(x='isp', data=df_nodes, ax=ax[0], palette='magma')
        ax[0].set_title('Phishing (1) vs Non-Phishing (0) Distribution')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Count (Log Scale)')

    # 節點度數分佈圖 (Degree Distribution)
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

    print(f"\n1. Loading Ethereum dataset {ver}...")

    data = torch.load(fname, weights_only=False)
    print(data.train_mask.sum()) 
    print(data.test_mask.sum())
    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    
    return data



def visualize_blockchain_communities(G, partition, labels=None, node=2):

    G_sub = G
    
    # 2. 準備佈局 (Layout)
    # k 控制節點間距，iterations 增加佈局穩定度
    pos = nx.spring_layout(G_sub, k=0.15, seed=42, iterations=50)
    
    # 3. 準備顏色與數據
    community_ids = [partition.get(node, -1) for node in G_sub.nodes()]
    num_communities = len(set(community_ids))
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 4. 繪製邊 (設定透明度以免干擾視覺)
    nx.draw_networkx_edges(G_sub, pos, edge_color='gray', ax=ax)
    # nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    
    # 5. 繪製節點 (依據社群 ID 上色)
    im = nx.draw_networkx_nodes(
        G_sub, pos, 
        node_size=80, 
        node_color=community_ids, 
        cmap='jet', 
        ax=ax
    )
    
    # 6. [進階創新] 高亮顯示異常節點 (用紅色外框或 X 標記)
    if labels is not None:
        # 假設 labels 中 1 代表 illicit (異常)
        illicit_nodes = [n for n in G_sub.nodes() if labels.get(n) == 1]
        nx.draw_networkx_nodes(
            G_sub, pos, 
            nodelist=illicit_nodes,
            node_size=120,
            node_color='none',
            edgecolors='red', # 紅色外框
            linewidths=2,
            label='Illicit Nodes',
            ax=ax
        )

    # 7. 細節調整
    plt.colorbar(im, label='Community ID')
    ax.set_title(f"Blockchain Community Detection (Detected {num_communities} Communities)")
    ax.axis('off')
    
    # 加上 Legend (如果 labels 存在)
    if labels is not None:
        plt.legend(scatterpoints=1)
    
    save_path=f"crime/crime_graph_analysis_{node}.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig) # 釋放記憶體
    print(f"✅ 可視化結果已儲存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: e1 or e2')
    args = parser.parse_args()
    if args.dataset == "e1":
        process_elliptic_data()
    
    elif args.dataset == "e1nol":
        process_elliptic_data_noL()
    elif args.dataset == "e1G":
        data = load_elliptic_data()
        G_nx = to_networkx(data, to_undirected=True)
        node_labels = {i: int(data.y[i].item()) for i in range(data.num_nodes)}
        louvain_features, partition = get_louvain_features(data.edge_index, data.num_nodes, data.y, data.train_mask)   
        illicit_nodes = [n for n in G_nx.nodes() if node_labels.get(n) == 1]
        if len(illicit_nodes) > 0:
            for node in range(len(illicit_nodes)):
                # 我們選取第一個非法節點作為中心
                target_node = illicit_nodes[node] 
                print(f"Generating Ego Graph for Illicit Node: {target_node}")
                # 抓取 2 層鄰居
                ego_nodes = nx.single_source_shortest_path_length(G_nx, target_node, cutoff=2).keys()
                G_ego = G_nx.subgraph(ego_nodes)
                visualize_blockchain_communities(G_ego, partition=partition, labels=node_labels, node=node)
                print("done")

    if args.dataset == "e2":
        process_ethereum_data()
    elif args.dataset == "e2nol":
        process_ethereum_data_noL()
#  
# eda_ethereum()
# subgraph_process()