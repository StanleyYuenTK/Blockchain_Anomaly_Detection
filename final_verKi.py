# data preprocessing
import os
import numpy as np
import pandas as pd
import random
import networkx as nx
import traceback
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree, get_ppr
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from community import community_louvain

# model
from sklearn.ensemble import IsolationForest
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, GIN
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, MLP,
    APPNP, ChebConv, GCN2Conv, MixHopConv
)
from catboost import CatBoostClassifier


# Optimization
import optuna
from optuna.samplers import TPESampler
import pygad

# visualiz
from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations
from GNNs import APPNPModel

RANDOM_SEED = 24027277
# {
#     'accuracy': 0,
#     'f1': 0,
#     'precision': 0,
#     'recall': 0,
#     'auc': 0,
#     'gmean': 0,
#     'macro_f1': 0,
#     'macro_precision': 0,
#     'macro_recall': 0,
#     'macro_auc': 0,
# }



# ==============================================================================
# 1. Data loading
# ==============================================================================
def load_elliptic_data(dataset_dir='Dataset'):

    # 1. load data
    classes_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_classes.csv'))
    edgelist_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv'))
    features_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_features.csv'), header=None)

    # 2. colA is id, colB is time steps
    features_df.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(2, features_df.shape[1])]
    
    # 3. labelled class 2 (licit), labelled class 1 (illicit), unknown -> -1
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    y = torch.tensor(nodes_df['class'].map({'2': 0, '1': 1}).fillna(-1).values, dtype=torch.long)
    
    # 4. 提取特徵 
    x = torch.tensor(nodes_df.iloc[:, 1:-1].values, dtype=torch.float)

    # 5. 高效處理邊表 (取代 iterrows)
    tx_id_map = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'])}
    
    # 使用 map 進行向量化轉換，速度提升顯著
    edge_index_src = edgelist_df.iloc[:, 0].map(tx_id_map)
    edge_index_tgt = edgelist_df.iloc[:, 1].map(tx_id_map)
    
    # 移除不在 map 中的無效邊 (dropna)
    edges = pd.concat([edge_index_src, edge_index_tgt], axis=1).dropna().astype(int)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # 6. 構建數據集與 Mask
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)

    known_mask = (y != -1)
    data.train_mask = (data.timesteps < 35) & known_mask
    data.val_mask   = (data.timesteps >= 35) & (data.timesteps < 42) & known_mask
    data.test_mask  = (data.timesteps >= 42) & known_mask

    print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

    return data

# ==============================================================================
# 2. Feature Engineering
# ==============================================================================

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
    edges = edge_index.t().cpu().numpy()
    G.add_edges_from(edges)
    
    # Run Louvain community detection
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=RANDOM_SEED)
    
    # Get community IDs for all nodes
    comm_ids = np.array([partition.get(i, -1) for i in range(num_nodes)])
    
    # Compute community statistics
    comm_size = {}
    comm_train_illicit = {}
    comm_train_total = {}
    
    labels_np = labels.cpu().numpy() if labels is not None else np.zeros(num_nodes)
    train_mask_np = train_mask.cpu().numpy() if train_mask is not None else np.ones(num_nodes, dtype=bool)
    
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


def get_neighbor_loader(data, batch_size=1024, num_neighbors=[25, 10], shuffle=True):
    # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/multi_gpu_vanilla.html
    train_indices = torch.where(data.train_mask)[0]
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_indices,
        shuffle=shuffle
    )

     


# ==============================================================================
# Genetic Algorithm
# https://pygad.readthedocs.io/en/latest/
# ==============================================================================

# ==============================================================================
# ALL Model
# ==============================================================================

def isolation_forest_baseline(data):
    
    X_train = data.x[data.train_mask].cpu().numpy()
    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    # Train Isolation Forest
    clf = IsolationForest(random_state=24027277)
    clf.fit(X_train)

    # Predict: 1=normal, -1=anomaly
    y_pred = clf.predict(X_test)
    anomaly_scores = clf.decision_function(X_test)

    # Convert: 1=normal, -1=anomaly -> 1=anomaly, 0=normal
    y_pred = (y_pred == -1).astype(int)

    # Evaluate
    baseline_results = {
        'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'macro_auc': roc_auc_score(y_test, -anomaly_scores),
        'gmean': np.sqrt(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 
                        (1 - precision_score(y_test, y_pred, pos_label=0, zero_division=0))),
        'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'auc': roc_auc_score(y_test, -anomaly_scores),
        'accuracy': accuracy_score(y_test, y_pred),
    }
    return baseline_results


def gnn_train(model, data, epochs=100, lr=0.01):
    neighbor_loader = get_neighbor_loader(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in neighbor_loader:
            batch = batch.to(data.x.device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                out = model(batch.x, batch.edge_index)
                pred = out.argmax(dim=1)
                train_acc = (pred[batch.train_mask] == batch.y[batch.train_mask]).sum().item() / batch.train_mask.sum().item()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}")
            model.train()
    return model

def gnn_test(model, data): 
    test_mask = data.test_mask
    y = data.y

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_test = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        
        # Calculate metrics
        test_acc = (pred[test_mask] == y[test_mask]).sum().item() / test_mask.sum().item()
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        out_probs = torch.softmax(out, dim=1)
        auc = roc_auc_score(y_test, out_probs[test_mask, 1].cpu().numpy())
        
        return {
            'accuracy': test_acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
        }
            


def gnn_train_and_test(model_name, data, 
        in_channels=None, hidden_channels=64, out_channels=2, 
        num_layers=2, dropout=0.5, epochs=100, lr=0.01, heads=8
    ):

    in_channels = data.x.size(1)
    # Initialize GCN model
    if model_name == 'GCN':
        model = GCN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm').to(data.x.device)
    elif model_name == 'GAT':
        model = GAT(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', heads=heads).to(data.x.device)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', aggr='mean').to(data.x.device)    
    elif model_name == 'GIN':
        model = GIN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm').to(data.x.device)
    elif model_name == 'APPNP':
        model = APPNPModel(in_channels, hidden_channels, out_channels).to(data.x.device)
        
    model = gnn_train(model, data)
    results = gnn_test(model, data)
    
    
    print(f"\n{model_name} Test Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    return results


# ==============================================================================
# Main - Execute complete GNN anomaly detection  
# ==============================================================================
def main():
    print("=" * 60)
    print("Blockchain Anomaly Detection GNN Framework")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # 1. Load Elliptic dataset
    # ========================================================================
    print("\n1. Loading Elliptic dataset...")
    elliptic_data = load_elliptic_data()
    elliptic_data = elliptic_data.to(device)
    print(f"Data loaded: {elliptic_data.x.size(0)} nodes, {elliptic_data.x.size(1)} features, {elliptic_data.edge_index.size(1)} edges")

    # ========================================================================
    # 2. Feature Engineering, pagerank, degree, louvain
    # ========================================================================
    print("\n2. Feature Engineering...")
    
    print("PageRank features...")
    pagerank_features = get_pagerank_features(elliptic_data.edge_index.cpu(), elliptic_data.x.size(0)).to(device)
    
    print("Degree features...")
    degree_features = get_degree_features(elliptic_data.edge_index.cpu(), elliptic_data.x.size(0)).to(device)
    
    # print("Louvain features...")
    # louvain_features, partition = get_louvain_features(elliptic_data.edge_index.cpu(), elliptic_data.x.size(0), labels=elliptic_data.y, train_mask=elliptic_data.train_mask)
    # louvain_features = louvain_features.to(device)

    # elliptic_data.x = torch.cat([elliptic_data.x, pagerank_features, degree_features, louvain_features], dim=1)
    elliptic_data.x = torch.cat([elliptic_data.x, pagerank_features, degree_features], dim=1)
    print(f"Features Engineering done...\nTotal features: {elliptic_data.x.size(1)} dimensions")
    
    # Apply StandardScaler to normalize all features
    print("\nStandardScaler...")
    x_numpy = elliptic_data.x.cpu().numpy()  # Convert to numpy
    x_scaled = StandardScaler().fit_transform(x_numpy)  # Fit and transform
    elliptic_data.x = torch.tensor(x_scaled, dtype=torch.float).to(elliptic_data.x.device)  # Convert back to tensor
    print(f"StandardScaler done...\nTotal features: {elliptic_data.x.size(1)} dimensions")
    

    # ========================================================================
    # 3. Train Isolation Forest baseline - done - 暫時comment for train GNN model
    # ========================================================================
    # print("\n3. Training Isolation Forest baseline...")
    # baseline_results = isolation_forest_baseline(elliptic_data)
    # print(f"Isolation Forest baseline results:")
    # for key, value in baseline_results.items():
    #     print(f"{key}: {value}")
    # print("Baseline evaluation completed")

    # ========================================================================
    # 4. Train GCN model
    # ========================================================================
    print("\n4. Training GNNs baseline...")

    # gnn_models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII',
    #     'MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN',
    #     'MixHop_GCN_K3', 'MixHop_GAT_K3', 'MixHop_GraphSAGE_K3', 'MixHop_GIN_K3',
    #     'MixHop_GCN_K4', 'MixHop_GAT_K4', 'MixHop_GraphSAGE_K4', 'MixHop_GIN_K4',
    # ]

    gnn_models_list = ['APPNP', 'GCN', 'GAT', 'GraphSAGE', 'GIN']

    for model_name in gnn_models_list:
        print(f"\n--- Training {model_name} ---")
        results = gnn_train_and_test(model_name=model_name, data=elliptic_data)
        # print(f"GCN results: {results}")
        print("GNN training completed")


if __name__ == "__main__":
    main()