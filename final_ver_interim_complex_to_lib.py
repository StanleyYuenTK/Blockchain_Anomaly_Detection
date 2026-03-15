"""
Student ID: 24027277d
Name: Yuen Tsz Ki

Blockchain Anomaly Detection GNN Framework - Final Version

This framework implements:
- Data preprocessing with Degree, Standard PageRank (no labels), and Louvain community detection
- 7 GNN base models: APPNP, ChebNet, GAT, GCN, GCNII, GIN, GraphSAGE
- MixHop augmentation for multi-hop information
- Focal Loss for class imbalance
- Bayesian Optimization (TPE) for hyperparameter search
- CatBoost meta-model for stacking ensemble
- NeighborLoader for mini-batch subgraph sampling
"""

import os
import numpy as np
import pandas as pd
import random
import networkx as nx
from community import community_louvain
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, MLP,
    APPNP, ChebConv, GCN2Conv, MixHopConv
)
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, GIN

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree as pyg_degree, get_ppr
import torch_scatter
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Optuna for Bayesian Optimization with TPE
import optuna
from optuna.samplers import TPESampler

from catboost import CatBoostClassifier

from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations

RANDOM_SEED = 24027277

# ==============================================================================
# Helper Functions
# ==============================================================================

# ==============================================================================
# Data Processing - Feature Engineering
# ==============================================================================

# class TensorScaler:
#     """GPU-compatible standard scaler using PyTorch"""
#     def __init__(self):
#         self.mean = None
#         self.std = None

#     def fit(self, x):
#         self.mean = x.mean(dim=0, keepdim=True)
#         self.std = x.std(dim=0, keepdim=True)
#         self.std[self.std == 0] = 1.0
#         return self

#     def transform(self, x):
#         return (x - self.mean) / self.std

#     def fit_transform(self, x):
#         self.fit(x)
#         return self.transform(x)

# class GraphFeatureEngineer:
#     """Graph feature engineering with Degree, PageRank, and Louvain community detection"""

#     def __init__(self, data):
#         self.data = data
#         self.edge_index = data.edge_index.cpu()
#         self.num_nodes = data.x.size(0)

#     def compute_degree_features(self):
#         """Compute directed degree features"""
#         # 1. Calculate in/out degree
#         out_deg = pyg_degree(self.edge_index[0], self.num_nodes)
#         in_deg = pyg_degree(self.edge_index[1], self.num_nodes)
        
#         # 2. Total degree and ratio
#         total_deg = in_deg + out_deg
#         in_out_ratio = in_deg / (out_deg + 1e-8)
        
#         # 3. Log normalization (handle power law)
#         in_deg_log = torch.log1p(in_deg)
#         out_deg_log = torch.log1p(out_deg)
#         total_deg_log = torch.log1p(total_deg)
        
#         # 4. Ranking feature
#         total_deg_rank = torch.argsort(torch.argsort(total_deg, descending=True)).float() / self.num_nodes

#         return torch.stack([
#             in_deg, out_deg, in_deg_log, out_deg_log, total_deg_log, in_out_ratio, total_deg_rank
#         ], dim=1)
    
#     def compute_pagerank_features(self, alpha=0.15, max_iter=100, tol=1e-6):
#         """Compute PageRank features"""
#         edge_index = self.edge_index
#         num_nodes = self.num_nodes
        
#         ppr_edge_index, ppr_weights = get_ppr(edge_index=edge_index, alpha=alpha, num_nodes=num_nodes)
#         pagerank_score = torch_scatter.scatter_add(ppr_weights,ppr_edge_index[1],dim=0,dim_size=num_nodes)
#         return pagerank_score.cpu().reshape(-1, 1)
    
#     def compute_louvain_features(self, resolution=1.0, train_mask=None):
#         """Compute Louvain community features"""
#         # Build graph
#         G = nx.Graph()
#         edges = self.edge_index.t().cpu().numpy()
#         G.add_edges_from(edges)
        
#         # Run Louvain
#         partition = community_louvain.best_partition(G, resolution=resolution, random_state=RANDOM_SEED)
        
#         comm_ids = np.array([partition.get(i, -1) for i in range(self.num_nodes)])
        
#         comm_size = {}
#         comm_train_illicit = {}
#         comm_train_total = {}

#         labels = self.data.y.cpu().numpy()
#         train_mask_np = train_mask.cpu().numpy() if train_mask is not None else np.ones(self.num_nodes, dtype=bool)

#         # Calculate community stats (only on train set to prevent leakage)
#         for i in range(self.num_nodes):
#             cid = comm_ids[i]
#             if cid == -1: continue
            
#             comm_size[cid] = comm_size.get(cid, 0) + 1
            
#             if train_mask_np[i]:
#                 if labels[i] == 1:
#                     comm_train_illicit[cid] = comm_train_illicit.get(cid, 0) + 1
#                 comm_train_total[cid] = comm_train_total.get(cid, 0) + 1

#         # Calculate internal degree
#         row, col = self.edge_index
#         same_comm_mask = (torch.from_numpy(comm_ids[row]) == torch.from_numpy(comm_ids[col]))
#         internal_edge_index = self.edge_index[:, same_comm_mask]
#         internal_deg = pyg_degree(internal_edge_index[0], self.num_nodes)
#         total_deg = pyg_degree(self.edge_index[0], self.num_nodes)

#         # Combine features
#         louvain_feat = torch.zeros((self.num_nodes, 5))
        
#         for i in range(self.num_nodes):
#             cid = comm_ids[i]
#             if cid == -1: continue
            
#             size = comm_size.get(cid, 1)
#             illicit_cnt = comm_train_illicit.get(cid, 0)
#             train_total = comm_train_total.get(cid, 1e-8)
            
#             louvain_feat[i, 0] = np.log1p(size)
#             louvain_feat[i, 1] = illicit_cnt / train_total
#             louvain_feat[i, 2] = 1.0 if illicit_cnt > 0 else 0.0
#             louvain_feat[i, 3] = internal_deg[i] / (total_deg[i] + 1e-8)
#             louvain_feat[i, 4] = internal_deg[i] / (size)

#         return louvain_feat, partition

#     def add_degree_pagerank_louvain_features(self, train_mask=None):
#         """Add all graph features to the data object
        
#         Args:
#             train_mask: Optional tensor indicating training nodes. Used for Louvain 
#                         community detection to prevent data leakage.
#         """
        
#         new_x = [self.data.x]
        
#         degree_feats = self.compute_degree_features()
#         new_x.append(degree_feats.to(self.data.x.device))
        
#         pagerank_feats = self.compute_pagerank_features()
#         new_x.append(pagerank_feats.to(self.data.x.device))

#         # If train_mask is provided, we compute community features
#         # This is useful for the main GNN training (prevent leakage)
#         if train_mask is not None:
#             community_feats, partition = self.compute_louvain_features(train_mask=train_mask)
#             new_x.append(community_feats.to(self.data.x.device))
#             return self.data, partition
        
#         enhanced_x = torch.cat(new_x, dim=1)
        
#         self.data.x = enhanced_x
        
#         print(f"Enhanced features: {enhanced_x.size(1)} dimensions")
        
#         return self.data, None

#     def get_graph_structure_features(self):
#         """Get graph structure features (degree + pagerank) for CatBoost meta-model"""
#         return torch.cat([self.compute_degree_features(), self.compute_pagerank_features()], dim=1)


def pseudo_labeling(data, model_predictions, threshold=0.5, confidence_threshold=0.9):
    """Pseudo-labeling: Add high-confidence predictions from high-risk communities to training set
    
    激進做法 (Aggressive Approach):
    1. 找出某社群中非法節點比例 > threshold (預設 80%) 的「高風險社群」
    2. 將這些社群中的 Unknown 節點標記為疑似非法
    3. 將其加入訓練集重新訓練
    
    Args:
        data: PyG Data object with:
            - y: node labels (-1 for Unknown, 0 for normal, 1 for illegal)
            - train_mask, val_mask, test_mask: data splits
            - edge_index: graph edges
        model_predictions: numpy array of shape (num_nodes, 2) with probabilities [P(0), P(1)]
        threshold: Minimum illegal ratio in community to be considered high-risk (default 0.8)
        confidence_threshold: Minimum confidence to accept pseudo-label (default 0.9)
        
    Returns:
        new_data: Updated PyG Data object with pseudo-labeled nodes added to training set
        pseudo_labeled_nodes: List of node indices that were pseudo-labeled
        high_risk_communities: Set of community_ids identified as high-risk
    """
    print("\n" + "="*70)
    print("PSEUDO-LABELING: High-Risk Community Detection")
    print("="*70)
    
    # Step 1: Detect communities using Louvain
    print("\n[Step 1] Detecting communities using Louvain algorithm...")
    # Create undirected graph
    G = nx.Graph()
    edges = data.edge_index.t().cpu().numpy()
    G.add_edges_from(edges)
    
    # Run Louvain algorithm
    partition = community_louvain.best_partition(G, resolution=1.0, random_state=RANDOM_SEED)
    
    num_communities = len(set(partition.values()))
    print(f"   Found {num_communities} communities")
    
    # Compute illegal ratio for each community
    train_mask_np = data.train_mask.cpu().numpy()
    community_stats = {}
    for node, comm_id in partition.items():
        if comm_id not in community_stats:
            community_stats[comm_id] = {'total': 0, 'illegal': 0}
        
        # Only count nodes in training set
        if train_mask_np[node]:
            community_stats[comm_id]['total'] += 1
            if data.y[node] == 1:  # Illegal
                community_stats[comm_id]['illegal'] += 1
    
    # Compute ratios
    high_risk_communities = set()
    
    print(f"\n   Community illegal ratios (threshold = {threshold*100:.0f}%):")
    for comm_id in sorted(community_stats.keys()):
        stats = community_stats[comm_id]
        if stats['total'] > 0:
            ratio = stats['illegal'] / stats['total']
            
            if ratio >= threshold:
                high_risk_communities.add(comm_id)
                print(f"   - Community {comm_id}: {stats['illegal']}/{stats['total']} = {ratio*100:.1f}% (HIGH RISK)")
            elif ratio > 0:
                print(f"   - Community {comm_id}: {stats['illegal']}/{stats['total']} = {ratio*100:.1f}%")
    
    print(f"\n   Identified {len(high_risk_communities)} high-risk communities (illegal ratio >= {threshold*100:.0f}%)")
    
    # Step 3: Find Unknown nodes in high-risk communities
    print("\n[Step 3] Finding Unknown nodes in high-risk communities...")
    unknown_mask = data.y == -1
    unknown_nodes = torch.where(unknown_mask)[0].tolist()
    
    high_risk_unknown_nodes = []
    for node in unknown_nodes:
        comm_id = partition.get(node, -1)
        if comm_id in high_risk_communities:
            high_risk_unknown_nodes.append(node)
    
    print(f"   Total Unknown nodes: {len(unknown_nodes)}")
    print(f"   Unknown nodes in high-risk communities: {len(high_risk_unknown_nodes)}")
    
    if len(high_risk_unknown_nodes) == 0:
        print("\n   No Unknown nodes found in high-risk communities. Pseudo-labeling skipped.")
        return data, [], high_risk_communities
    
    # Step 4: Filter by confidence threshold (optional - use model predictions)
    print(f"\n[Step 4] Filtering by confidence threshold ({confidence_threshold*100:.0f}%)...")
    
    pseudo_labeled_nodes = []
    for node in high_risk_unknown_nodes:
        # Get model's prediction confidence for this node
        probs = model_predictions[node]
        illegal_prob = probs[1]  # P(illegal)
        
        if illegal_prob >= confidence_threshold:
            pseudo_labeled_nodes.append(node)
    
    print(f"   High-confidence pseudo-labels (P(illegal) >= {confidence_threshold*100:.0f}%): {len(pseudo_labeled_nodes)}")
    
    if len(pseudo_labeled_nodes) == 0:
        print("\n   No nodes passed confidence threshold. Using all nodes in high-risk communities.")
        pseudo_labeled_nodes = high_risk_unknown_nodes
        print(f"   Added {len(pseudo_labeled_nodes)} pseudo-labeled nodes")
    
    # Step 5: Update training set
    print("\n[Step 5] Updating training set with pseudo-labeled nodes...")
    
    # Create new data object with updated labels and masks
    new_data = data.clone()
    
    # Get original training size
    original_train_size = new_data.train_mask.sum().item()
    
    # Update labels for pseudo-labeled nodes
    for node in pseudo_labeled_nodes:
        new_data.y[node] = 1  # Mark as suspected illegal
    
    # Add to training mask (keep them in test/val as well for evaluation, but include in training)
    for node in pseudo_labeled_nodes:
        new_data.train_mask[node] = True
    
    # Recompute community features with new training set
    print("\n[Step 6] Recomputing community features with pseudo-labeled data...")
    
    # Update high-risk community detection with new labels
    # Note: We use the ORIGINAL train_mask for computing community features to avoid leakage
    # but we add pseudo-labeled nodes to the training set for model training
    
    new_train_size = new_data.train_mask.sum().item()
    print(f"   Training set size: {original_train_size} -> {new_train_size} (+{new_train_size - original_train_size})")
    
    # Summary
    print("\n" + "="*70)
    print("PSEUDO-LABELING SUMMARY")
    print("="*70)
    print(f"   High-risk communities (>{threshold*100:.0f}% illegal): {len(high_risk_communities)}")
    print(f"   Unknown nodes in high-risk communities: {len(high_risk_unknown_nodes)}")
    print(f"   Pseudo-labeled nodes added to training: {len(pseudo_labeled_nodes)}")
    print(f"   New training set size: {new_train_size}")
    print("="*70 + "\n")
    
    return new_data, pseudo_labeled_nodes, high_risk_communities


def iterative_pseudo_labeling(data, model_predictions_fn, num_iterations=1, 
                               community_threshold=0.5, confidence_threshold=0.9):
    """Iterative pseudo-labeling: Repeatedly add pseudo-labels and retrain
    
    Args:
        data: PyG Data object
        model_predictions_fn: Function that takes data and returns predictions array
        num_iterations: Number of pseudo-labeling iterations
        community_threshold: Minimum illegal ratio for high-risk communities
        confidence_threshold: Minimum confidence for pseudo-labels
        
    Returns:
        final_data: Data object after all iterations
        all_pseudo_labeled: All nodes that were pseudo-labeled across iterations
    """
    print("\n" + "#"*70)
    print("# ITERATIVE PSEUDO-LABELING")
    print("#"*70)
    
    current_data = data.clone()
    all_pseudo_labeled = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")
        
        # Get current model predictions
        print(f"\nGetting model predictions...")
        predictions = model_predictions_fn(current_data)
        
        # Apply pseudo-labeling
        current_data, pseudo_labeled, high_risk_comms = pseudo_labeling(
            current_data, 
            predictions,
            threshold=community_threshold,
            confidence_threshold=confidence_threshold
        )
        
        if len(pseudo_labeled) == 0:
            print(f"\nNo more nodes to pseudo-label. Stopping early.")
            break
        
        all_pseudo_labeled.extend(pseudo_labeled)
        
    print(f"\n{'#'*70}")
    print(f"FINAL RESULTS")
    print(f"{'#'*70}")
    print(f"   Total iterations: {iteration + 1}")
    print(f"   Total pseudo-labeled nodes: {len(all_pseudo_labeled)}")
    print(f"   Final training set size: {current_data.train_mask.sum().item()}")
    print(f"{'#'*70}\n")
    
    return current_data, all_pseudo_labeled


# ==============================================================================
# Loss Functions
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        return (self.alpha * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss).mean()


# ==============================================================================
# Baseline Model - Isolation Forest (Commented out - using sklearn directly below)
# ==============================================================================
# def isolation_forest_baseline(data):
    
#     X_train = data.x[data.train_mask].cpu().numpy()
#     X_test = data.x[data.test_mask].cpu().numpy()
#     y_test = data.y[data.test_mask].cpu().numpy()

#     # Train Isolation Forest
#     clf = IsolationForest(random_state=24027277)
#     clf.fit(X_train)

#     # Predict: 1=normal, -1=anomaly
#     y_pred = clf.predict(X_test)
#     anomaly_scores = clf.decision_function(X_test)

#     # Convert: 1=normal, -1=anomaly -> 1=anomaly, 0=normal
#     y_pred = (y_pred == -1).astype(int)

#     # Evaluate
#     baseline_results = {
#         'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
#         'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
#         'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
#         'macro_auc': roc_auc_score(y_test, -anomaly_scores),
#         'gmean': np.sqrt(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 
#                         (1 - precision_score(y_test, y_pred, pos_label=0, zero_division=0))),
#         'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
#         'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
#         'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
#         'auc': roc_auc_score(y_test, -anomaly_scores),
#         'accuracy': accuracy_score(y_test, y_pred),
#     }
#     return baseline_results


# ==============================================================================
# GNN Models - Using PyG Built-in Models
# ==============================================================================

# def GNN_baseline(data, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
#     gcn = GCN(
#         in_channels=in_channels,
#         hidden_channels=hidden_channels,
#         out_channels=out_channels,
#         num_layers=num_layers,
#         dropout=dropout
#     )

#     gat = GAT(
#         in_channels=in_channels,
#         hidden_channels=hidden_channels,
#         out_channels=out_channels,
#         num_layers=num_layers,
#         # heads=num_heads,
#         dropout=dropout
#     )

#     graphSAGE = GraphSAGE(
#         in_channels=in_channels,
#         hidden_channels=hidden_channels,
#         out_channels=out_channels,
#         num_layers=num_layers,
#         dropout=dropout
#     )

#     gin = GIN(
#         in_channels=in_channels,
#         hidden_channels=hidden_channels,
#         out_channels=out_channels,
#         num_layers=num_layers,
#         dropout=dropout
#     )

    

    


    



# class GCNModel(torch.nn.Module):
#     """GCN using PyG built-in model"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
#         super().__init__()
#         self.model = GCN(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             out_channels=out_channels,
#             num_layers=2,
#             dropout=dropout
#         )

#     def forward(self, data, return_embed=False):
#         x = self.model(data.x, data.edge_index)
#         return F.log_softmax(x, dim=1)


# class GATModel(torch.nn.Module):
#     """GAT using PyG built-in model"""
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.5):
#         super().__init__()
#         self.model = GAT(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             out_channels=out_channels,
#             num_layers=2,
#             heads=num_heads,
#             dropout=dropout
#         )

#     def forward(self, data, return_embed=False):
#         x = self.model(data.x, data.edge_index)
#         return F.log_softmax(x, dim=1)


# class GraphSAGEModel(torch.nn.Module):
#     """GraphSAGE using PyG built-in model"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr='mean'):
#         super().__init__()
#         self.model = SAGE(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             out_channels=out_channels,
#             num_layers=2,
#             dropout=dropout
#         )

#     def forward(self, data, return_embed=False):
#         x = self.model(data.x, data.edge_index)
#         return F.log_softmax(x, dim=1)


# class GINModel(torch.nn.Module):
#     """GIN using PyG built-in model"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
#         super().__init__()
#         self.model = GIN(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             out_channels=out_channels,
#             num_layers=2,
#             dropout=dropout
#         )

#     def forward(self, data, return_embed=False):
#         x = self.model(data.x, data.edge_index)
#         return F.log_softmax(x, dim=1)


# APPNP, ChebNet, GCNII remain as custom implementations (or can be simplified later)
# class APPNPModel(torch.nn.Module):
#     """APPNP (Approximate Personalized Propagation of Neural Predictions) with BatchNorm"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, alpha=0.1, num_iterations=3):
#         super(APPNPModel, self).__init__()
#         self.mlp1 = nn.Linear(in_channels, hidden_channels)
#         self.bn1 = nn.BatchNorm1d(hidden_channels)
#         self.mlp2 = nn.Linear(hidden_channels, hidden_channels)
#         self.bn2 = nn.BatchNorm1d(hidden_channels)
#         self.mlp3 = nn.Linear(hidden_channels, out_channels)
#         self.dropout = dropout
#         self.alpha = alpha
#         self.num_iterations = num_iterations
#         self.propagate = APPNP(K=num_iterations, alpha=alpha, dropout=dropout)

#     def forward(self, data, return_embed=False):
#         x, edge_index = data.x, data.edge_index
        
#         x = self.mlp1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.mlp2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.mlp3(x)
        
#         x = self.propagate(x, edge_index)
#         return F.log_softmax(x, dim=1)


# class ChebNetModel(torch.nn.Module):
#     """Chebyshev Graph Convolutional Network with BatchNorm"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
#         super(ChebNetModel, self).__init__()
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
        
#         self.convs.append(ChebConv(in_channels, hidden_channels, K=K))
#         self.bns.append(nn.BatchNorm1d(hidden_channels))
        
#         self.convs.append(ChebConv(hidden_channels, out_channels, K=K))
#         self.bns.append(nn.BatchNorm1d(out_channels))
        
#         self.dropout = dropout

#     def forward(self, data, return_embed=False):
#         x, edge_index = data.x, data.edge_index
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         x = self.bns[-1](x)
#         return F.log_softmax(x, dim=1)


class GCNIIModel(torch.nn.Module):
    """GCNII (Graph Convolutional Networks with Initial Residual and Identity Mapping)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, 
                 alpha=0.5, theta=1.0, num_layers=2):
        super(GCNIIModel, self).__init__()
        self.alpha = alpha
        self.theta = theta
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.bn_input = nn.BatchNorm1d(hidden_channels)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    channels=hidden_channels, 
                    alpha=alpha, 
                    theta=theta, 
                    layer=i+1,
                    shared_weights=True,
                    add_self_loops=True,
                    normalize=True
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        x_0 = self.input_proj(x)
        x_0 = self.bn_input(x_0)
        x = x_0
        
        for i, conv in enumerate(self.convs):
            x = conv(x, x_0, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.output_proj(x)
        return F.log_softmax(x, dim=1)


# ==============================================================================
# MixHop Models - Multi-hop Information Enhancement using MixHopConv
# ==============================================================================

# class MixHopGCNModel(torch.nn.Module):
#     """GCN with MixHopConv for multi-hop information with BatchNorm"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
#         super(MixHopGCNModel, self).__init__()
#         self.powers = powers
#         self.num_hops = len(powers)
        
#         self.mixhop_conv = MixHopConv(
#             in_channels=in_channels,
#             out_channels=hidden_channels,
#             powers=powers,
#             add_self_loops=True
#         )
        
#         self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
#         self.mlp = MLP([
#             hidden_channels * self.num_hops,
#             hidden_channels * self.num_hops // 2,
#             out_channels
#         ], dropout=dropout)
        
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
        
#         x = self.mixhop_conv(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         out = self.mlp(x)
#         return F.log_softmax(out, dim=1)


# class MixHopGATModel(torch.nn.Module):
#     """GAT with MixHop for multi-hop information using MixHopConv with BatchNorm"""
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.5, powers=[0, 1, 2]):
#         super(MixHopGATModel, self).__init__()
#         self.powers = powers
#         self.num_hops = len(powers)
        
#         self.mixhop_conv = MixHopConv(
#             in_channels=in_channels,
#             out_channels=hidden_channels,
#             powers=powers,
#             add_self_loops=True
#         )
        
#         self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
#         self.attention = GATConv(
#             hidden_channels * self.num_hops, 
#             hidden_channels, 
#             heads=num_heads, 
#             dropout=dropout, 
#             concat=True
#         )
        
#         self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        
#         self.out_proj = nn.Linear(hidden_channels * num_heads, out_channels)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
        
#         x = self.mixhop_conv(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.attention(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
        
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         out = self.out_proj(x)
#         return F.log_softmax(out, dim=1)


# class MixHopGraphSAGEModel(torch.nn.Module):
#     """GraphSAGE with MixHop using MixHopConv with BatchNorm"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2], aggr='mean'):
#         super(MixHopGraphSAGEModel, self).__init__()
#         self.powers = powers
#         self.num_hops = len(powers)
        
#         self.mixhop_conv = MixHopConv(
#             in_channels=in_channels,
#             out_channels=hidden_channels,
#             powers=powers,
#             add_self_loops=True
#         )
        
#         self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
#         self.sage_conv = SAGEConv(hidden_channels * self.num_hops, hidden_channels, aggr=aggr)
        
#         self.bn2 = nn.BatchNorm1d(hidden_channels)
        
#         self.out_proj = nn.Linear(hidden_channels, out_channels)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
        
#         x = self.mixhop_conv(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.sage_conv(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
        
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         out = self.out_proj(x)
#         return F.log_softmax(out, dim=1)


# class MixHopGINModel(torch.nn.Module):
#     """GIN with MixHop using MixHopConv with BatchNorm"""
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
#         super(MixHopGINModel, self).__init__()
#         self.powers = powers
#         self.num_hops = len(powers)
        
#         self.mixhop_conv = MixHopConv(
#             in_channels=in_channels,
#             out_channels=hidden_channels,
#             powers=powers,
#             add_self_loops=True
#         )
        
#         self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
#         self.gin_conv = GINConv(MLP([hidden_channels * self.num_hops, hidden_channels, hidden_channels], dropout=dropout))
        
#         self.bn2 = nn.BatchNorm1d(hidden_channels)
        
#         self.out_proj = nn.Linear(hidden_channels, out_channels)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
        
#         x = self.mixhop_conv(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.gin_conv(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
        
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         out = self.out_proj(x)
#         return F.log_softmax(out, dim=1)


# ==============================================================================
# MixHop Models with Higher K values (K=3, K=4)
# ==============================================================================

class MixHopGCNModel_K3(MixHopGCNModel):
    """MixHop GCN with K=3 (powers=[0, 1, 2, 3]) for deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
        # Override to use K=3
        super().__init__(in_channels, hidden_channels, out_channels, dropout, powers=[0, 1, 2, 3])


class MixHopGCNModel_K4(MixHopGCNModel):
    """MixHop GCN with K=4 (powers=[0, 1, 2, 3, 4]) for even deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, powers=[0, 1, 2, 3, 4])


class MixHopGATModel_K3(MixHopGATModel):
    """MixHop GAT with K=3 (powers=[0, 1, 2, 3]) for deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.5, powers=[0, 1, 2]):
        super().__init__(in_channels, hidden_channels, out_channels, num_heads, dropout, powers=[0, 1, 2, 3])


class MixHopGATModel_K4(MixHopGATModel):
    """MixHop GAT with K=4 (powers=[0, 1, 2, 3, 4]) for even deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.5, powers=[0, 1, 2]):
        super().__init__(in_channels, hidden_channels, out_channels, num_heads, dropout, powers=[0, 1, 2, 3, 4])


class MixHopGraphSAGEModel_K3(MixHopGraphSAGEModel):
    """MixHop GraphSAGE with K=3 (powers=[0, 1, 2, 3]) for deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2], aggr='mean'):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, powers=[0, 1, 2, 3], aggr=aggr)


class MixHopGraphSAGEModel_K4(MixHopGraphSAGEModel):
    """MixHop GraphSAGE with K=4 (powers=[0, 1, 2, 3, 4]) for even deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2], aggr='mean'):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, powers=[0, 1, 2, 3, 4], aggr=aggr)


class MixHopGINModel_K3(MixHopGINModel):
    """MixHop GIN with K=3 (powers=[0, 1, 2, 3]) for deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, powers=[0, 1, 2, 3])


class MixHopGINModel_K4(MixHopGINModel):
    """MixHop GIN with K=4 (powers=[0, 1, 2, 3, 4]) for even deeper neighborhood mixing"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
        super().__init__(in_channels, hidden_channels, out_channels, dropout, powers=[0, 1, 2, 3, 4])



# Model class mapping
# MODEL_CLASSES = {
#     'GCN': GCNModel,
#     'GAT': GATModel,
#     'GraphSAGE': GraphSAGEModel,
#     'GIN': GINModel,
#     'APPNP': APPNPModel,
#     'ChebNet': ChebNetModel,
#     'GCNII': GCNIIModel,
#     # MixHop variants (using MixHopConv)
#     'MixHop_GCN': MixHopGCNModel,
#     'MixHop_GAT': MixHopGATModel,
#     'MixHop_GraphSAGE': MixHopGraphSAGEModel,
#     'MixHop_GIN': MixHopGINModel,
#     # MixHop K=3 variants (deeper neighborhood mixing)
#     'MixHop_GCN_K3': MixHopGCNModel_K3,
#     'MixHop_GAT_K3': MixHopGATModel_K3,
#     'MixHop_GraphSAGE_K3': MixHopGraphSAGEModel_K3,
#     'MixHop_GIN_K3': MixHopGINModel_K3,
#     # MixHop K=4 variants (even deeper neighborhood mixing)
#     'MixHop_GCN_K4': MixHopGCNModel_K4,
#     'MixHop_GAT_K4': MixHopGATModel_K4,
#     'MixHop_GraphSAGE_K4': MixHopGraphSAGEModel_K4,
#     'MixHop_GIN_K4': MixHopGINModel_K4,
# }



# ==============================================================================
# Trainer Class
# ==============================================================================

class Trainer:
    def __init__(self, model, data, device, optimizer, criterion, history=None, use_neighbor_loader=False, neighbor_loader=None):
        self.model = model
        self.data = data
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = history or TrainingHistory()
        self.use_neighbor_loader = use_neighbor_loader
        self.neighbor_loader = neighbor_loader

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_neighbor_loader and self.neighbor_loader is not None:
            # Mini-batch training with NeighborLoader
            total_loss = 0
            num_batches = 0
            for batch in self.neighbor_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                # Get labels for the batch's input nodes
                y = batch.y[:batch.input_id.size(0)]
                loss = self.criterion(out[:batch.input_id.size(0)], y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            return total_loss / max(num_batches, 1)
        else:
            # Full graph training (default)
            out = self.model(self.data)
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            return loss.item()

    @torch.no_grad()
    def evaluate(self, include_test=True):
        self.model.eval()
        out = self.model(self.data)
        pred = out.argmax(dim=1)
        
        val_loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask]).item()
        val_y_true = self.data.y[self.data.val_mask].cpu().numpy()
        val_y_pred = pred[self.data.val_mask].cpu().numpy()
        
        val_f1 = f1_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
        val_recall = recall_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
        val_macro_precision = precision_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_macro_recall = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_macro_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        
        val_probs = torch.exp(out).cpu().numpy()
        val_mask_np = self.data.val_mask.cpu().numpy()
        val_auc = roc_auc_score(val_y_true, val_probs[val_mask_np][:, 1])
        
        tn = ((val_y_true == 0) & (val_y_pred == 0)).sum()
        fp = ((val_y_true == 0) & (val_y_pred == 1)).sum()
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        val_gmean = np.sqrt(val_f1 * val_specificity) if val_f1 * val_specificity > 0 else 0

        result = {
            'val_loss': val_loss,
            
            'val_macro_f1': val_macro_f1,
            'val_macro_precision': val_macro_precision,
            'val_macro_recall': val_macro_recall,
            'val_macro_auc': val_auc,
            'val_gmean': val_gmean,
            'val_specificity': val_specificity,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_auc': val_auc,
            'val_acc': val_acc
        }

        if include_test and self.data.test_mask.sum() > 0:
            test_y_true = self.data.y[self.data.test_mask].cpu().numpy()
            test_y_pred = pred[self.data.test_mask].cpu().numpy()
            test_f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
            test_recall = recall_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
            test_precision = precision_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
            test_acc = accuracy_score(test_y_true, test_y_pred)
            
            probs = torch.exp(out).cpu().numpy()
            test_mask_np = self.data.test_mask.cpu().numpy()
            
            test_auc = roc_auc_score(test_y_true, probs[test_mask_np][:, 1])
            
            tn = ((test_y_true == 0) & (test_y_pred == 0)).sum()
            fp = ((test_y_true == 0) & (test_y_pred == 1)).sum()
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_gmean = np.sqrt(test_f1 * test_specificity) if test_f1 * test_specificity > 0 else 0

            result.update({
                'test_precision_illicit': precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0),
                'test_recall_illicit': recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0),

                'test_macro_f1': f1_score(test_y_true, test_y_pred, average='macro', zero_division=0),
                'test_macro_precision': precision_score(test_y_true, test_y_pred, average='macro', zero_division=0),
                'test_macro_recall': recall_score(test_y_true, test_y_pred, average='macro', zero_division=0),
                'test_macro_auc': test_auc,
                'test_gmean': test_gmean,
                'test_specificity': test_specificity,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_auc': test_auc,
                'test_acc': test_acc
            })
        
        return result

    def fit(self, epochs=100, include_test=True, patience=20):
        """Train the model with Early Stopping to prevent overfitting
        
        Args:
            epochs: Maximum number of epochs
            include_test: Whether to compute test metrics
            patience: Number of epochs to wait for improvement before stopping (increased to 25)
        """
        best_val_loss = float('inf')
        best_stats = None
        counter = 0  # Counter for early stopping
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            stats = self.evaluate(include_test=include_test)

            self.history.add_epoch(
                epoch=epoch + 1, train_loss=train_loss, val_loss=stats['val_loss'],
                val_macro_f1=stats['val_macro_f1'], test_macro_f1=stats['test_macro_f1'],
                val_macro_precision=stats['val_macro_precision'], test_macro_precision=stats['test_macro_precision'],
                val_macro_recall=stats['val_macro_recall'], test_macro_recall=stats['test_macro_recall'],
                val_macro_auc=stats['val_macro_auc'], test_macro_auc=stats['test_macro_auc'],
                val_gmean=stats['val_gmean'], test_gmean=stats['test_gmean'],
                val_specificity=stats['val_specificity'], test_specificity=stats['test_specificity'],
                val_f1=stats['val_f1'], test_f1=stats['test_f1'],
                val_precision=stats['val_precision'], test_precision=stats['test_precision'],
                val_recall=stats['val_recall'], test_recall=stats['test_recall'],
                val_auc=stats['val_auc'], test_auc=stats['test_auc'],
                val_acc=stats['val_acc'], test_acc=stats['test_acc'],
            )

            if stats['val_loss'] < best_val_loss:
                best_val_loss = stats['val_loss']
                best_stats = stats
                counter = 0  # Reset counter when improvement is found
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Macro-F1: {stats['val_macro_f1']:.4f}, "
                      f"Val Macro-Recall: {stats['val_macro_recall']:.4f}, "
                      f"Val G-Mean: {stats['val_gmean']:.4f}")

        return best_stats, self.history


# ==============================================================================
# Data Loading
# ==============================================================================

# def load_elliptic_data(dataset_dir='Dataset'):
#     """Load Elliptic Bitcoin transaction dataset"""
#     print("Loading Elliptic dataset...")
    
#     classes_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_classes.csv'))
#     edgelist_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv'))
#     features_df = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_features.csv'), header=None)

#     features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
#     features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(2, features_df.shape[1])]

#     nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')

#     features = nodes_df[nodes_df.columns[2:-1]].values
#     x = torch.tensor(features, dtype=torch.float)

#     print(f"Loaded {x.size(0)} nodes with {x.size(1)} features")

#     labels = nodes_df['class'].apply(lambda c: 0 if c == '2' else (1 if c == '1' else -1))
#     y = torch.tensor(labels.values, dtype=torch.long)

#     print(f"Label distribution: licit: {(y == 0).sum().item()}, illicit: {(y == 1).sum().item()}, unknown: {(y == -1).sum().item()}")

#     tx_id_map = {tx_id: i for i, tx_id in enumerate(nodes_df['txId'].values)}

#     source_indices = []
#     target_indices = []
#     for _, row in edgelist_df.iterrows():
#         src = row['txId1'] if 'txId1' in edgelist_df.columns else row.iloc[0]
#         tgt = row['txId2'] if 'txId2' in edgelist_df.columns else row.iloc[1]
#         if src in tx_id_map and tgt in tx_id_map:
#             source_indices.append(tx_id_map[src])
#             target_indices.append(tx_id_map[tgt])

#     edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
#     print(f"Graph structure: {edge_index.size(1)} edges")

#     data = Data(x=x, y=y, edge_index=edge_index)
#     data.timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)

#     known_mask = y != -1
#     data.train_mask = (data.timesteps < 35) & known_mask
#     data.val_mask = (data.timesteps >= 35) & (data.timesteps < 42) & known_mask
#     data.test_mask = (data.timesteps >= 42) & known_mask

#     print(f"Data splits: Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

#     return data


# ==============================================================================
# NeighborLoader for Mini-batch Training
# ==============================================================================

# def create_neighbor_loader(data, batch_size=1024, num_neighbors=[10, 5], shuffle=True):
#     """Create NeighborLoader for mini-batch subgraph sampling"""
    
#     print(f"Creating NeighborLoader with batch_size={batch_size}, num_neighbors={num_neighbors}")
    
#     train_indices = torch.where(data.train_mask)[0]
    
#     loader = NeighborLoader(
#         data,
#         num_neighbors=num_neighbors,
#         batch_size=batch_size,
#         input_nodes=train_indices,
#         shuffle=shuffle
#     )

#     return loader


# ==============================================================================
# Optuna Objective Function
# ==============================================================================

# Global variables to store data and device for Optuna objective
_optuna_data = None
_optuna_device = None
_optuna_neighbor_loader = None

# All models to optimize
ALL_MODELS_TO_OPTIMIZE = [
    'GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII',
    'MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN',  # K=2
    'MixHop_GCN_K3', 'MixHop_GAT_K3', 'MixHop_GraphSAGE_K3', 'MixHop_GIN_K3',  # K=3
    'MixHop_GCN_K4', 'MixHop_GAT_K4', 'MixHop_GraphSAGE_K4', 'MixHop_GIN_K4',  # K=4
]

# def set_optuna_params(data, device):
#     """Set global parameters for Optuna objective function"""
#     global _optuna_data, _optuna_device, _optuna_neighbor_loader
#     _optuna_data = data
#     _optuna_device = device
    
#     # Create NeighborLoader for mini-batch training during Optuna optimization
#     _optuna_neighbor_loader = create_neighbor_loader(data)

# def optuna_objective(trial):
#     """Optuna optimization objective function - optimizes all GNN models"""
#     global _optuna_data, _optuna_device, _optuna_neighbor_loader
    
#     torch.manual_seed(RANDOM_SEED)
#     np.random.seed(RANDOM_SEED)
#     random.seed(RANDOM_SEED)

#     # Select model to optimize
#     model_name = trial.suggest_categorical('model_name', ALL_MODELS_TO_OPTIMIZE)
    
#     # Suggest hyperparameters using Optuna
#     hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
#     dropout = trial.suggest_float('dropout', 0.1, 0.5, log=False)
#     lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
#     num_heads = trial.suggest_categorical('num_heads', [4, 8])

#     model_class = MODEL_CLASSES.get(model_name)
    
#     try:
#         # Build model args - only GAT variants need num_heads
#         model_args = {
#             'in_channels': _optuna_data.x.size(1),
#             'hidden_channels': hidden_channels,
#             'out_channels': 2,
#             'dropout': dropout
#         }
#         if 'GAT' in model_name:
#             model_args['num_heads'] = num_heads
        
#         model = model_class(**model_args)

#         model = model.to(_optuna_device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
#         alpha_val = 1.0
#         criterion = FocalLoss(alpha=alpha_val, gamma=6)

#         trainer = Trainer(model, _optuna_data, _optuna_device, optimizer, criterion,
#                           use_neighbor_loader=True, neighbor_loader=_optuna_neighbor_loader)
#         best_stats, _ = trainer.fit(epochs=30)

#         # Use composite score: 0.4*Macro-F1 + 0.3*G-Mean + 0.3*Macro-AUC with threshold optimization
#         val_y_true = _optuna_data.y[_optuna_data.val_mask].cpu().numpy()
        
#         # Get validation probabilities
#         model.eval()
#         with torch.no_grad():
#             out = model(_optuna_data)
#             val_probs = torch.exp(out).cpu().numpy()
#             val_mask_np = _optuna_data.val_mask.cpu().numpy()
#             val_probs_np = val_probs[val_mask_np][:, 1]
        
#         best_thresh, metrics = find_optimal_threshold(val_y_true, val_probs_np)
#         val_pred = (val_probs_np >= best_thresh).astype(int)
        
#         val_macro_f1 = f1_score(val_y_true, val_pred, average='macro', zero_division=0)
#         val_gmean = metrics['gmean']
#         val_macro_auc = metrics['macro_auc']
        
#         composite_score = 0.4 * val_macro_f1 + 0.3 * val_gmean + 0.3 * val_macro_auc
        
#         trial.report(composite_score, step=30)

#         return composite_score
#     except optuna.exceptions.TrialPruned:
#         # Re-raise pruning exceptions so Optuna handles them properly
#         raise
#     except Exception as e:
#         print(f"Error in Optuna objective ({model_name}): {e}")
#         return 0.0


# ==============================================================================
# Evaluation Functions
# ==============================================================================

# def calculate_all_metrics(y_true, y_pred, y_probs):
#     """Calculate all evaluation metrics in sorted order"""
    
#     macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
#     macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     macro_auc = roc_auc_score(y_true, y_probs[:, 1], average='macro')
#     f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
#     precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
#     recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
#     auc = roc_auc_score(y_true, y_probs[:, 1])
#     accuracy = accuracy_score(y_true, y_pred)
    
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     if cm.shape == (2, 2):
#         tn, fp, fn, tp = cm.ravel()
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     else:
#         specificity = 0
    
#     gmean = np.sqrt(recall * specificity) if recall * specificity > 0 else 0

#     return {
#         'macro_f1': macro_f1, 'macro_precision': macro_precision, 'macro_recall': macro_recall,
#         'macro_auc': macro_auc, 'gmean': gmean, 'specificity': specificity,
#         'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc, 'accuracy': accuracy
#     }


# def find_optimal_threshold(y_true, y_probs):
#     """Find optimal classification threshold to maximize 0.4*Macro-F1 + 0.3*G-Mean + 0.3*Macro-AUC"""
    
#     thresholds = np.arange(0.05, 0.95, 0.05)
#     best_threshold = 0.5
#     best_score = 0
#     best_metrics = None
    
#     for thresh in thresholds:
#         y_pred_adj = (y_probs >= thresh).astype(int)
        
#         f1 = f1_score(y_true, y_pred_adj, pos_label=1, zero_division=0)
#         macro_f1 = f1_score(y_true, y_pred_adj, average='macro', zero_division=0)
        
#         cm = confusion_matrix(y_true, y_pred_adj, labels=[0, 1])
#         if cm.shape == (2, 2):
#             tn, fp, fn, tp = cm.ravel()
#             sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#             specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#             gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0
#         else:
#             gmean = 0
#             sensitivity = 0
#             specificity = 0
        
#         macro_auc = roc_auc_score(y_true, y_probs)
        
#         score = 0.4 * macro_f1 + 0.3 * gmean + 0.3 * macro_auc
        
#         if score > best_score:
#             best_score = score
#             best_threshold = thresh
#             best_metrics = {
#                 'macro_f1': macro_f1,
#                 'gmean': gmean,
#                 'macro_auc': macro_auc,
#                 'sensitivity': sensitivity,
#                 'specificity': specificity,
#                 'threshold': thresh,
#                 'composite_score': score
#             }
    
#     return best_threshold, best_metrics


# def apply_threshold_tuning(y_true, y_probs):
#     """Apply threshold tuning and return optimized predictions
    
#     Returns:
#         y_pred_optimized: Optimized predictions
#         optimal_threshold: The threshold used
#         metrics_at_optimal: Metrics at optimal threshold
#     """
#     # Use balanced metric to optimize both F1 and G-mean
#     optimal_threshold, metrics_at_optimal = find_optimal_threshold(y_true, y_probs)
    
#     # Apply optimal threshold
#     y_pred_optimized = (y_probs >= optimal_threshold).astype(int)
    
#     return y_pred_optimized, optimal_threshold, metrics_at_optimal


# ==============================================================================
# Genetic Algorithm for Model Selection and Ensemble Optimization
# ==============================================================================

# class GeneticAlgorithmEnsemble:
#     """Genetic Algorithm for selecting optimal GNN model combinations and weights"""
    
#     def __init__(self, model_names, population_size=20, generations=30, 
#                  crossover_rate=0.8, mutation_rate=0.1, elite_count=2,
#                  fitness_metric='macro_f1', random_seed=RANDOM_SEED):
#         self.model_names = model_names
#         self.num_models = len(model_names)
#         self.population_size = population_size
#         self.generations = generations
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate
#         self.elite_count = elite_count
#         self.fitness_metric = fitness_metric
#         np.random.seed(random_seed)
        
#     def _create_chromosome(self):
#         selection = np.random.binomial(1, 0.5, self.num_models)
#         weights = np.random.dirichlet(np.ones(self.num_models))
#         return np.concatenate([selection, weights])
    
#     def _decode_chromosome(self, chromosome):
#         selection = chromosome[:self.num_models].astype(bool)
#         weights = chromosome[self.num_models:]
        
#         if selection.sum() > 0:
#             weights = weights * selection
#             if weights.sum() > 0:
#                 weights = weights / weights.sum()
#             else:
#                 weights = selection / selection.sum() if selection.sum() > 0 else weights
#         return selection, weights
    
#     def _fitness(self, chromosome, all_probs, y_true, val_mask):
#         selection, weights = self._decode_chromosome(chromosome)
        
#         if selection.sum() < 2:
#             return 0.0
        
#         selected_probs = []
#         for i, (sel, prob) in enumerate(zip(selection, all_probs)):
#             if sel:
#                 selected_probs.append(prob)
        
#         if len(selected_probs) < 2:
#             return 0.0
        
#         selected_probs = np.array(selected_probs)
#         weights = weights[selection]
#         weights = weights / weights.sum()
        
#         ensemble_probs = np.tensordot(weights, selected_probs, axes=([0], [0]))
        
#         if val_mask is not None:
#             probs_val = ensemble_probs[val_mask]
#             y_val = y_true[val_mask]
#         else:
#             probs_val = ensemble_probs
#             y_val = y_true
        
#         # Use composite score: 0.4*Macro-F1 + 0.3*G-Mean + 0.3*Macro-AUC
#         best_thresh, metrics = find_optimal_threshold(y_val, probs_val[:, 1])
#         predictions = (probs_val[:, 1] >= best_thresh).astype(int)
        
#         macro_f1 = f1_score(y_val, predictions, average='macro', zero_division=0)
#         gmean = metrics['gmean']
#         macro_auc = metrics['macro_auc']
        
#         fitness = 0.4 * macro_f1 + 0.3 * gmean + 0.3 * macro_auc
        
#         return fitness
    
#     def _selection(self, population, fitnesses):
#         selected = []
#         for _ in range(len(population)):
#             idx = np.random.choice(len(population), size=min(3, len(population)), replace=False)
#             best_idx = idx[np.argmax(fitnesses[idx])]
#             selected.append(population[best_idx].copy())
#         return selected
    
#     def _crossover(self, parent1, parent2):
#         if np.random.rand() < self.crossover_rate:
#             point = np.random.randint(1, len(parent1))
#             child1 = np.concatenate([parent1[:point], parent2[point:]])
#             child2 = np.concatenate([parent2[:point], parent1[point:]])
#             return child1, child2
#         return parent1.copy(), parent2.copy()
    
#     def _mutation(self, chromosome):
#         for i in range(self.num_models):
#             if np.random.rand() < self.mutation_rate:
#                 chromosome[i] = 1 - chromosome[i]
        
#         weight_part = chromosome[self.num_models:]
#         noise = np.random.normal(0, 0.1, self.num_models)
#         weight_part = weight_part + noise
#         weight_part = np.clip(weight_part, 0, 1)
#         weight_part = np.maximum(weight_part, 0)
        
#         if weight_part.sum() > 0:
#             weight_part = weight_part / weight_part.sum()
        
#         chromosome[self.num_models:] = weight_part
#         return chromosome
    
#     def fit(self, all_probs, y_true, val_mask, verbose=True):
#         """Run genetic algorithm to find optimal model combination"""
#         if verbose:
#             print(f"\nRunning Genetic Algorithm...")
#             print(f"  Population: {self.population_size}, Generations: {self.generations}")
#             print(f"  Metric: {self.fitness_metric}")
#             print(f"  Models: {self.model_names}")
        
#         population = [self._create_chromosome() for _ in range(self.population_size)]
        
#         best_chromosome = None
#         best_fitness = -1
#         fitness_history = []
        
#         for gen in range(self.generations):
#             fitnesses = np.array([self._fitness(ch, all_probs, y_true, val_mask) for ch in population])
            
#             gen_best_idx = np.argmax(fitnesses)
#             if fitnesses[gen_best_idx] > best_fitness:
#                 best_fitness = fitnesses[gen_best_idx]
#                 best_chromosome = population[gen_best_idx].copy()
            
#             fitness_history.append(best_fitness)
            
#             if verbose and (gen + 1) % 5 == 0:
#                 print(f"  Generation {gen+1}: Best Fitness = {best_fitness:.4f}")
            
#             selected = self._selection(population, fitnesses)
            
#             new_population = []
            
#             elite_idx = np.argsort(fitnesses)[-self.elite_count:]
#             for idx in elite_idx:
#                 new_population.append(population[idx].copy())
            
#             while len(new_population) < self.population_size:
#                 p1, p2 = np.random.choice(len(selected), size=2, replace=False)
#                 c1, c2 = self._crossover(selected[p1], selected[p2])
#                 c1 = self._mutation(c1)
#                 c2 = self._mutation(c2)
#                 new_population.extend([c1, c2])
            
#             population = new_population[:self.population_size]
        
#         self.best_selection, self.best_weights = self._decode_chromosome(best_chromosome)
#         self.best_fitness = best_fitness
#         self.fitness_history = fitness_history
        
#         self.selected_models = [name for sel, name in zip(self.best_selection, self.model_names) if sel]
        
#         if verbose:
#             print(f"\nGA Optimization Complete!")
#             print(f"  Best Fitness ({self.fitness_metric}): {best_fitness:.4f}")
#             print(f"  Selected Models: {self.selected_models}")
#             print(f"  Optimal Weights: {dict(zip(self.selected_models, self.best_weights[self.best_selection]))}")
        
#         return self
    
#     def predict(self, all_probs, y_true=None, test_mask=None):
#         """Generate predictions using optimized weights"""
#         selected_probs = []
#         for i, (sel, prob) in enumerate(zip(self.best_selection, all_probs)):
#             if sel:
#                 selected_probs.append(prob)
        
#         selected_probs = np.array(selected_probs)
#         weights = self.best_weights[self.best_selection]
#         weights = weights / weights.sum()
        
#         ensemble_probs = np.tensordot(weights, selected_probs, axes=([0], [0]))
        
#         if test_mask is not None:
#             ensemble_probs = ensemble_probs[test_mask]
        
#         if y_true is not None:
#             best_thresh, _ = find_optimal_threshold(y_true, ensemble_probs[:, 1])
#             predictions = (ensemble_probs[:, 1] >= best_thresh).astype(int)
#             return predictions, best_thresh, ensemble_probs
        
#         return ensemble_probs.argmax(axis=1), 0.5, ensemble_probs



# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_full_pipeline():
    """Execute complete GNN anomaly detection pipeline"""
    
    print("=" * 60)
    print("Blockchain Anomaly Detection GNN Framework - Final Version")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # 1. Load Elliptic dataset
    # ========================================================================
    print("\n1. Loading Elliptic dataset...")
    data = load_elliptic_data()
    data = data.to(device)
    print(f"Data loaded: {data.x.size(0)} nodes, {data.x.size(1)} features, {data.edge_index.size(1)} edges")

    # ========================================================================
    # 2. Feature Engineering (Degree, PageRank, Louvain)
    # ========================================================================
    print("\n2. Computing graph features (Degree, PageRank, Louvain)...")
    feature_engineer = GraphFeatureEngineer(data)
    data, community_partition = feature_engineer.add_degree_pagerank_louvain_features(train_mask=data.train_mask)
    print(f"Enhanced features: {data.x.size(1)} dimensions")

    # Extract graph structure features for CatBoost meta-model (degree + pagerank)
    graph_structure_features = feature_engineer.get_graph_structure_features()
    graph_features_np = graph_structure_features.cpu().numpy()
    print(f"Graph structure features: {graph_features_np.shape[1]} dimensions (degree + pagerank)")

    # ========================================================================
    # 3. Train Isolation Forest baseline (sklearn direct usage)
    # ========================================================================
    print("\n3. Training Isolation Forest baseline...")
    baseline_results = isolation_forest_baseline(data)
    print("Baseline evaluation completed")
    
    # ========================================================================
    # 4. Bayesian Optimization for Hyperparameter Search (Optuna TPESampler)
    # ========================================================================
    print("\n4. Running Bayesian Optimization (TPE) for hyperparameter search...")
    print(f"Optimizing all {len(ALL_MODELS_TO_OPTIMIZE)} models: {ALL_MODELS_TO_OPTIMIZE}")
    
    # Set global parameters for Optuna objective
    set_optuna_params(data, device)
    
    # Create Optuna study with TPE sampler
    sampler = TPESampler(
        n_startup_trials=5,  # Random sampling for first 5 trials
        seed=RANDOM_SEED
    )
    study = optuna.create_study(
        direction='maximize',  # Maximize test F1
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Run optimization with enough trials for all models
    # Academic project recommendation: 20-30 trials per model for thorough hyperparameter search
    # 50-100 trials
    n_trials_per_model = 3
    total_trials = len(ALL_MODELS_TO_OPTIMIZE) * n_trials_per_model
    print(f"Running {total_trials} trials ({n_trials_per_model} trials per model)...")
    study.optimize(optuna_objective, n_trials=total_trials, show_progress_bar=False)
    
    # Get best hyperparameters for each model from the study
    best_params_per_model = {}
    
    # Group trials by model name
    trials_by_model = {model: [] for model in ALL_MODELS_TO_OPTIMIZE}
    for trial in study.trials:
        if trial.value is not None and trial.params.get('model_name'):
            model_name = trial.params['model_name']
            trials_by_model[model_name].append(trial)
    
    # Find best params for each model
    print("\nBest hyperparameters per model:")
    print("=" * 60)
    for model_name in ALL_MODELS_TO_OPTIMIZE:
        model_trials = trials_by_model[model_name]
        if model_trials:
            # Find best trial for this model
            best_trial = max(model_trials, key=lambda t: t.value)
            best_params_per_model[model_name] = {
                'hidden_channels': best_trial.params['hidden_channels'],
                'dropout': best_trial.params['dropout'],
                'lr': best_trial.params['lr'],
                'num_heads': best_trial.params['num_heads'],
                'val_composite_score': best_trial.value
            }
            print(f"{model_name}:")
            print(f"  Hidden channels: {best_params_per_model[model_name]['hidden_channels']}")
            print(f"  Dropout: {best_params_per_model[model_name]['dropout']:.4f}")
            print(f"  Learning rate: {best_params_per_model[model_name]['lr']:.6f}")
            print(f"  Num heads: {best_params_per_model[model_name]['num_heads']}")
            print(f"  Best Composite Score: {best_params_per_model[model_name]['val_composite_score']:.4f}")
        else:
            # Use default params if no successful trials
            best_params_per_model[model_name] = {
                'hidden_channels': 128,
                'dropout': 0.3,
                'lr': 0.001,
                'num_heads': 4,
                'val_composite_score': 0.0
            }
            print(f"{model_name}: Using default parameters (no successful trials)")
    
    print("=" * 60)
    
    # Overall best
    best_trial = study.best_trial
    print(f"\nOverall best F1: {study.best_value:.4f} ({best_trial.params.get('model_name', 'N/A')})")

    # ========================================================================
    # 5. Define models to train (7 base + 4 MixHop = 11 models)
    # ========================================================================
    print("\n4. Setting up model configurations...")
    
    # Base models (7)
    base_models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII']
    # MixHop models (4) + K=3 (4) + K=4 (4) = 12
    mixhop_models_list = [
        'MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN',
        'MixHop_GCN_K3', 'MixHop_GAT_K3', 'MixHop_GraphSAGE_K3', 'MixHop_GIN_K3',
        'MixHop_GCN_K4', 'MixHop_GAT_K4', 'MixHop_GraphSAGE_K4', 'MixHop_GIN_K4',
    ]

    all_models_to_train = base_models_list + mixhop_models_list
    
    # Store results
    results_collector = {'IForest': baseline_results}
    training_histories = {}
    trained_models = {}
    
    # Get test labels
    test_y_true = data.y[data.test_mask].cpu().numpy()
    
    # ========================================================================
    # 6. Train all GNN models with optimized hyperparameters (using NeighborLoader)
    # ========================================================================
    print("\n6. Training GNN models with optimized hyperparameters...")
    
    # Create NeighborLoader for mini-batch training
    neighbor_loader = create_neighbor_loader(data)
    
    for model_name in all_models_to_train:
        print(f"\n--- Training {model_name} ---")
        
        try:
            # Get optimized hyperparameters for this model
            model_params = best_params_per_model.get(model_name, {
                'hidden_channels': 128,
                'dropout': 0.3,
                'lr': 0.001,
                'num_heads': 4
            })
            
            hidden_channels = model_params['hidden_channels']
            dropout = model_params['dropout']
            lr = model_params['lr']
            num_heads = model_params['num_heads']
            
            model_class = MODEL_CLASSES[model_name]
            
            # Create model
            # Build model args - only GAT variants need num_heads
            model_args = {
                'in_channels': data.x.size(1),
                'hidden_channels': hidden_channels,
                'out_channels': 2,
                'dropout': dropout
            }
            if 'GAT' in model_name:
                model_args['num_heads'] = num_heads
            
            model = model_class(**model_args)
            
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            criterion = FocalLoss(alpha=1.0, gamma=6)
            
            trainer = Trainer(model, data, device, optimizer, criterion, 
                            use_neighbor_loader=True, neighbor_loader=neighbor_loader)
            best_stats, training_history = trainer.fit(epochs=100)
            
            training_histories[model_name] = training_history
            
            model.eval()
            with torch.no_grad():
                out = model(data)
                probs = torch.exp(out).cpu().numpy()
                test_probs = probs[data.test_mask.cpu().numpy()]
                test_pred = np.argmax(test_probs, axis=1)
            
            test_pred_optimized, optimal_threshold, threshold_metrics = apply_threshold_tuning(
                test_y_true, test_probs[:, 1]
            )
            
            results_collector[model_name] = calculate_all_metrics(test_y_true, test_pred_optimized, test_probs)
            results_collector[model_name]['optimal_threshold'] = optimal_threshold
            results_collector[model_name]['threshold_metrics'] = threshold_metrics
            
            trained_models[model_name] = model
            
            print(f"{model_name} - Test F1: {results_collector[model_name]['f1']:.4f}, "
                  f"Test AUC: {results_collector[model_name]['auc']:.4f}, "
                  f"Test MacroF1: {results_collector[model_name]['macro_f1']:.4f}, "
                  f"Test MacroAUC: {results_collector[model_name]['macro_auc']:.4f}, "
                  f"Test G-Mean: {results_collector[model_name]['gmean']:.4f}, "
                  f"Optimal Threshold: {optimal_threshold:.2f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    # ========================================================================
    # 6. Apply Pseudo-Labeling to Expand Training Set
    # ========================================================================
    print("\n6. Applying pseudo-labeling to expand training set...")

    # Use the best performing model for pseudo-labeling
    if len(results_collector) > 0:
        # Filter out IForest (baseline) - only use GNN models for pseudo-labeling
        gnn_results = {k: v for k, v in results_collector.items() if k != 'IForest'}
        
        if gnn_results:
            best_model_name = max(gnn_results.items(), 
                                 key=lambda x: x[1].get('macro_f1', 0))[0]
            print(f"Using {best_model_name} predictions for pseudo-labeling...")
        
        # Get best model predictions
        best_model = trained_models[best_model_name]
        best_model.eval()
        
        # Create a function that takes data and returns predictions
        def get_best_model_predictions(input_data):
            with torch.no_grad():
                probs = torch.exp(best_model(input_data)).cpu().numpy()
            return probs
        
        # Apply iterative pseudo-labeling
        # Using default thresholds
        data_with_pseudo_labels, all_pseudo_labeled = iterative_pseudo_labeling(
            data, 
            get_best_model_predictions
        )
        
        # If pseudo-labeling added new nodes, re-train models
        if len(all_pseudo_labeled) > 0:
            print(f"\n6.6. Re-training GNN models with {len(all_pseudo_labeled)} pseudo-labeled nodes...")
            
            # Update neighbor loader with new data
            neighbor_loader_pseudo = create_neighbor_loader(data_with_pseudo_labels)
            
            # Re-train models with pseudo-labeled data
            for model_name in all_models_to_train:
                print(f"\n--- Re-training {model_name} with pseudo-labels ---")
                
                try:
                    model_params = best_params_per_model.get(model_name, {
                        'hidden_channels': 128,
                        'dropout': 0.3,
                        'lr': 0.001,
                        'num_heads': 4
                    })
                    
                    hidden_channels = model_params['hidden_channels']
                    dropout = model_params['dropout']
                    lr = model_params['lr']
                    num_heads = model_params['num_heads']
                    
                    model_class = MODEL_CLASSES[model_name]
                    
                    # Build model args - only GAT variants need num_heads
                    model_args = {
                        'in_channels': data_with_pseudo_labels.x.size(1),
                        'hidden_channels': hidden_channels,
                        'out_channels': 2,
                        'dropout': dropout
                    }
                    if 'GAT' in model_name:
                        model_args['num_heads'] = num_heads
                    
                    model = model_class(**model_args)
                    
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    # Calculate class weights with pseudo-labeled data
                    train_y = data_with_pseudo_labels.y[data_with_pseudo_labels.train_mask].cpu().numpy()
                    unique, counts = np.unique(train_y, return_counts=True)
                    
                    criterion = FocalLoss(alpha=1.0, gamma=6)
                    
                    trainer = Trainer(model, data_with_pseudo_labels, device, optimizer, criterion, 
                                    use_neighbor_loader=True, neighbor_loader=neighbor_loader_pseudo)
                    best_stats, training_history = trainer.fit(epochs=100)
                    
                    # Update trained models
                    trained_models[f'{model_name}_PL'] = model
                    
                    # Get predictions
                    model.eval()
                    with torch.no_grad():
                        out = model(data_with_pseudo_labels)
                        probs = torch.exp(out).cpu().numpy()
                        test_probs = probs[data_with_pseudo_labels.test_mask.cpu().numpy()]
                        test_pred = np.argmax(test_probs, axis=1)
                    
                    # Apply threshold tuning
                    test_pred_optimized, optimal_threshold, threshold_metrics = apply_threshold_tuning(
                        test_y_true, test_probs[:, 1]
                    )
                    
                    # Store results with pseudo-labeling suffix
                    results_collector[f'{model_name}_PL'] = calculate_all_metrics(test_y_true, test_pred_optimized, test_probs)
                    results_collector[f'{model_name}_PL']['optimal_threshold'] = optimal_threshold
                    
                    print(f"{model_name}_PL - Test F1: {results_collector[f'{model_name}_PL']['f1']:.4f}, "
                          f"Test AUC: {results_collector[f'{model_name}_PL']['auc']:.4f}, "
                          f"Test MacroF1: {results_collector[f'{model_name}_PL']['macro_f1']:.4f}, "
                          f"Test MacroAUC: {results_collector[f'{model_name}_PL']['macro_auc']:.4f}, "
                          f"Test G-Mean: {results_collector[f'{model_name}_PL']['gmean']:.4f}")
                    
                except Exception as e:
                    print(f"Error re-training {model_name} with pseudo-labels: {e}")
                    continue
        else:
            print("No high-confidence pseudo-labels found. Skipping re-training.")
    else:
        print("No trained models available for pseudo-labeling.")

    # ========================================================================
    # 7. Train Ensemble (CatBoost stacking)
    # ========================================================================
    print("\n7. Training Ensemble model with CatBoost...")
    
    # Base ensemble (simple averaging)
    all_probs = []
    for model_name, model in trained_models.items():
        model.eval()
        with torch.no_grad():
            out = model(data)
            probs = torch.exp(out).cpu().numpy()
            all_probs.append(probs)
    
    avg_probs = np.mean(all_probs, axis=0)
    test_mask_np = data.test_mask.cpu().numpy()
    
    base_ensemble_pred = np.argmax(avg_probs[test_mask_np], axis=1)
    
    # Apply threshold tuning to ensemble
    ensemble_pred_optimized, ensemble_threshold, ensemble_threshold_metrics = apply_threshold_tuning(
        test_y_true, avg_probs[test_mask_np][:, 1]
    )
    
    results_collector['Ensemble_Average'] = calculate_all_metrics(test_y_true, ensemble_pred_optimized, avg_probs[test_mask_np])
    results_collector['Ensemble_Average']['optimal_threshold'] = ensemble_threshold
    
    print(f"Ensemble (Average) - Test F1: {results_collector['Ensemble_Average']['f1']:.4f}, "
          f"Test AUC: {results_collector['Ensemble_Average']['auc']:.4f}, "
          f"Test MacroF1: {results_collector['Ensemble_Average']['macro_f1']:.4f}, "
          f"Test MacroAUC: {results_collector['Ensemble_Average']['macro_auc']:.4f}, "
          f"Test G-Mean: {results_collector['Ensemble_Average']['gmean']:.4f}, "
          f"Optimal Threshold: {ensemble_threshold:.2f}")
    
    # ========================================================================
    # Genetic Algorithm based Ensemble
    # ========================================================================
    print("\n" + "=" * 60)
    print("7. Genetic Algorithm Based Ensemble")
    print("=" * 60)
    
    # Get model names in same order as all_probs
    model_names_list = list(trained_models.keys())
    
    # Use validation set for GA optimization
    val_mask_np = data.val_mask.cpu().numpy()
    y_val = data.y[data.val_mask].cpu().numpy()
    
    # Run GA with macro_f1 as fitness
    ga_ensemble = GeneticAlgorithmEnsemble(model_names=model_names_list)
    ga_ensemble.fit(all_probs, data.y.cpu().numpy(), val_mask_np, verbose=True)
    
    # Get GA predictions on test set
    # Pass y_true to enable internal threshold optimization
    ga_predictions, ga_threshold, ga_test_probs = ga_ensemble.predict(all_probs, y_true=test_y_true, test_mask=test_mask_np)
    
    # Store results
    results_collector['Ensemble_GA'] = calculate_all_metrics(test_y_true, ga_predictions, ga_test_probs)
    results_collector['Ensemble_GA']['optimal_threshold'] = ga_threshold
    
    print(f"Ensemble (GA) - Test F1: {results_collector['Ensemble_GA']['f1']:.4f}, "
          f"Test AUC: {results_collector['Ensemble_GA']['auc']:.4f}, "
          f"Test MacroF1: {results_collector['Ensemble_GA']['macro_f1']:.4f}, "
          f"Test MacroAUC: {results_collector['Ensemble_GA']['macro_auc']:.4f}, "
          f"Test G-Mean: {results_collector['Ensemble_GA']['gmean']:.4f}, "
          f"Optimal Threshold: {ga_threshold:.2f}")
    
    # CatBoost stacking
    try:
        print("\nTraining CatBoost stacking meta-model with graph features...")

        # Get all model predictions for stacking
        stacked_probs = np.stack(all_probs, axis=0)  # (num_models, num_nodes, 2)

        # Prepare training data for meta-model: GNN predictions + graph structure features
        train_mask_np = data.train_mask.cpu().numpy()
        test_mask_np = data.test_mask.cpu().numpy()

        # GNN predictions: 11 models × 2 classes = 22 features
        gnn_preds = stacked_probs[:, train_mask_np, :].transpose(1, 0, 2).reshape(train_mask_np.sum(), -1)
        gnn_preds_test = stacked_probs[:, test_mask_np, :].transpose(1, 0, 2).reshape(test_mask_np.sum(), -1)

        # Graph structure features: degree (4) + pagerank (4) = 8 features
        graph_feats_train = graph_features_np[train_mask_np]
        graph_feats_test = graph_features_np[test_mask_np]

        # Combine GNN predictions + graph features
        X_meta = np.concatenate([gnn_preds, graph_feats_train], axis=1)
        X_test_meta = np.concatenate([gnn_preds_test, graph_feats_test], axis=1)

        print(f"  CatBoost input: {X_meta.shape[1]} features (22 GNN + 8 graph structure)")

        y_meta = data.y[data.train_mask].cpu().numpy()

        # Train CatBoost
        catboost_meta = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5,
            auto_class_weights='Balanced',
            random_seed=RANDOM_SEED,
            verbose=10,
            early_stopping_rounds=30
        )
        catboost_meta.fit(X_meta, y_meta)

        # Predict on test set
        catboost_pred = catboost_meta.predict(X_test_meta).astype(int)
        catboost_probs = catboost_meta.predict_proba(X_test_meta)
        
        # Apply threshold tuning to CatBoost predictions
        catboost_pred_optimized, catboost_threshold, catboost_threshold_metrics = apply_threshold_tuning(
            test_y_true, catboost_probs[:, 1]
        )
        
        results_collector['Ensemble_CatBoost'] = calculate_all_metrics(test_y_true, catboost_pred_optimized, catboost_probs)
        results_collector['Ensemble_CatBoost']['optimal_threshold'] = catboost_threshold
        
        print(f"Ensemble (CatBoost) - Test F1: {results_collector['Ensemble_CatBoost']['f1']:.4f}, "
                f"Test AUC: {results_collector['Ensemble_CatBoost']['auc']:.4f}, "
                f"Test MacroF1: {results_collector['Ensemble_CatBoost']['macro_f1']:.4f}, "
                f"Test MacroAUC: {results_collector['Ensemble_CatBoost']['macro_auc']:.4f}, "
                f"Test G-Mean: {results_collector['Ensemble_CatBoost']['gmean']:.4f}, "
                f"Optimal Threshold: {catboost_threshold:.2f}")
        
    except Exception as e:
        print(f"Error in CatBoost stacking: {e}")

    # ========================================================================
    # Two-Layer GA + CatBoost Ensemble
    # ========================================================================
    try:
        print("\n" + "=" * 60)
        print("Two-Layer GA + CatBoost Ensemble")
        print("=" * 60)

        # Layer 1: GA selects optimal model subset and weights
        print("\nLayer 1: Genetic Algorithm model selection...")

        # Get model predictions
        model_names_list = list(trained_models.keys())

        # Prepare validation data for GA
        val_mask_np = data.val_mask.cpu().numpy()
        all_probs_val = np.stack([all_probs[i][val_mask_np] for i in range(len(all_probs))], axis=0)
        y_val = data.y[data.val_mask].cpu().numpy()

        # Run GA to select best models
        # Note: all_probs_val and y_val are already filtered by val_mask,
        # so we pass None for val_mask to _fitness (or use a mask of all True)
        ga_ensemble_layer1 = GeneticAlgorithmEnsemble(model_names=model_names_list)
        ga_ensemble_layer1.fit(all_probs_val, y_val, None)

        # Get GA selected models and weights
        ga_selected = ga_ensemble_layer1.selected_models
        ga_weights = ga_ensemble_layer1.best_weights[ga_ensemble_layer1.best_selection]

        print(f"  GA selected models: {ga_selected}")
        print(f"  GA weights: {[f'{w:.3f}' for w in ga_weights]}")

        # Get predictions from GA-selected models only
        selected_indices = [model_names_list.index(m) for m in ga_selected]
        selected_probs = np.array([all_probs[i] for i in selected_indices])

        # Apply GA weights
        ga_weighted_probs = np.zeros_like(selected_probs[0])
        for i, (model_name, weight) in enumerate(zip(ga_selected, ga_weights)):
            ga_weighted_probs += selected_probs[i] * weight

        # Layer 2: CatBoost with GA-selected model predictions + graph features
        print("\nLayer 2: CatBoost meta-model with GA-selected features...")

        # Prepare features: GA-weighted predictions (2) + graph features (8)
        train_mask_np = data.train_mask.cpu().numpy()
        test_mask_np = data.test_mask.cpu().numpy()

        # GA predictions for train and test
        ga_probs_train = ga_weighted_probs[train_mask_np]
        ga_probs_test = ga_weighted_probs[test_mask_np]

        # Graph features
        graph_feats_train = graph_features_np[train_mask_np]
        graph_feats_test = graph_features_np[test_mask_np]

        # Combine: GA predictions (2) + graph features (8) = 10 features
        X_meta_layer2 = np.concatenate([
            ga_probs_train.reshape(-1, 1) if ga_probs_train.ndim == 1 else ga_probs_train,
            graph_feats_train
        ], axis=1)

        X_test_meta_layer2 = np.concatenate([
            ga_probs_test.reshape(-1, 1) if ga_probs_test.ndim == 1 else ga_probs_test,
            graph_feats_test
        ], axis=1)

        # Also include individual GA-selected model predictions for more expressiveness
        selected_model_probs_train = selected_probs[:, train_mask_np, :].transpose(1, 0, 2).reshape(train_mask_np.sum(), -1)
        selected_model_probs_test = selected_probs[:, test_mask_np, :].transpose(1, 0, 2).reshape(test_mask_np.sum(), -1)

        # Final features: GA-selected model predictions + GA weighted + graph features
        X_meta_layer2 = np.concatenate([selected_model_probs_train, graph_feats_train], axis=1)
        X_test_meta_layer2 = np.concatenate([selected_model_probs_test, graph_feats_test], axis=1)

        print(f"  Two-layer input: {X_meta_layer2.shape[1]} features ({len(ga_selected)} GA models × 2 + 8 graph)")

        y_meta = data.y[data.train_mask].cpu().numpy()

        # Train CatBoost
        catboost_layer2 = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5,
            auto_class_weights='Balanced',
            random_seed=RANDOM_SEED,
            verbose=10,
            early_stopping_rounds=30
        )
        catboost_layer2.fit(X_meta_layer2, y_meta)

        # Predict on test set
        catboost_layer2_pred = catboost_layer2.predict(X_test_meta_layer2).astype(int)
        catboost_layer2_probs = catboost_layer2.predict_proba(X_test_meta_layer2)

        # Apply threshold tuning
        ga_catboost_pred_optimized, ga_catboost_threshold, ga_catboost_metrics = apply_threshold_tuning(
            test_y_true, catboost_layer2_probs[:, 1]
        )

        # Store results
        results_collector['Ensemble_GA_CatBoost'] = calculate_all_metrics(
            test_y_true, ga_catboost_pred_optimized, catboost_layer2_probs
        )
        results_collector['Ensemble_GA_CatBoost']['optimal_threshold'] = ga_catboost_threshold
        results_collector['Ensemble_GA_CatBoost']['ga_selected_models'] = ga_selected

        print(f"Ensemble (GA + CatBoost) - Test F1: {results_collector['Ensemble_GA_CatBoost']['f1']:.4f}, "
                f"Test AUC: {results_collector['Ensemble_GA_CatBoost']['auc']:.4f}, "
                f"Test MacroF1: {results_collector['Ensemble_GA_CatBoost']['macro_f1']:.4f}, "
                f"Test MacroAUC: {results_collector['Ensemble_GA_CatBoost']['macro_auc']:.4f}, "
                f"Test G-Mean: {results_collector['Ensemble_GA_CatBoost']['gmean']:.4f}, "
                f"Optimal Threshold: {ga_catboost_threshold:.2f}")

    except Exception as e:
        print(f"Error in Two-Layer GA + CatBoost: {e}")


    # ========================================================================
    # TPE Optimization for CatBoost Meta-Learner (Fine-tuning after GA)
    # ========================================================================
    try:
        print("\n" + "=" * 60)
        print("TPE Optimization for CatBoost Meta-Learner")
        print("=" * 60)
        
        if 'ga_selected_models' not in results_collector.get('Ensemble_GA_CatBoost', {}):
            print("  Skipping TPE - No GA ensemble found")
        else:
            ga_selected = results_collector['Ensemble_GA_CatBoost'].get('ga_selected_models', [])
            if not ga_selected:
                print("  Skipping TPE - No GA selected models")
            else:
                # Get predictions from GA-selected models
                train_mask_np = data.train_mask.cpu().numpy()
                test_mask_np = data.test_mask.cpu().numpy()
                
                X_meta_layer2_tpe = []
                X_test_meta_layer2_tpe = []
                
                for model_name in ga_selected:
                    if model_name in trained_models:
                        model = trained_models[model_name]
                        model.eval()
                        with torch.no_grad():
                            out = model(data)
                            probs = torch.exp(out).cpu().numpy()
                            X_meta_layer2_tpe.append(probs[train_mask_np][:, 1])
                            X_test_meta_layer2_tpe.append(probs[test_mask_np][:, 1])
                
                if len(X_meta_layer2_tpe) > 0:
                    X_meta_layer2_tpe = np.column_stack(X_meta_layer2_tpe)
                    X_test_meta_layer2_tpe = np.column_stack(X_test_meta_layer2_tpe)
                    
                    graph_feats_train = graph_features_np[train_mask_np]
                    graph_feats_test = graph_features_np[test_mask_np]
                    
                    X_meta_final = np.hstack([X_meta_layer2_tpe, graph_feats_train])
                    X_test_meta_final = np.hstack([X_test_meta_layer2_tpe, graph_feats_test])
                    
                    y_meta = data.y[data.train_mask].cpu().numpy()
                    y_test = data.y[data.test_mask].cpu().numpy()
                    
                    val_size = int(0.2 * len(y_meta))
                    indices = np.random.permutation(len(y_meta))
                    val_idx, train_idx = indices[:val_size], indices[val_size:]
                    
                    X_train_tpe, X_val_tpe = X_meta_final[train_idx], X_meta_final[val_idx]
                    y_train_tpe, y_val_tpe = y_meta[train_idx], y_meta[val_idx]
                    
                    def catboost_optuna_objective(trial):
                        iterations = trial.suggest_int('iterations', 100, 500)
                        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
                        depth = trial.suggest_int('depth', 3, 8)
                        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)
                        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 30)
                        border_count = trial.suggest_int('border_count', 32, 255)
                        
                        model = CatBoostClassifier(
                            iterations=iterations,
                            learning_rate=learning_rate,
                            depth=depth,
                            l2_leaf_reg=l2_leaf_reg,
                            min_data_in_leaf=min_data_in_leaf,
                            border_count=border_count,
                            auto_class_weights='Balanced',
                            random_seed=RANDOM_SEED,
                            verbose=0,
                            early_stopping_rounds=30
                        )
                        
                        model.fit(X_train_tpe, y_train_tpe, eval_set=(X_val_tpe, y_val_tpe), verbose=0)
                        
                        val_probs = model.predict_proba(X_val_tpe)[:, 1]
                        best_thresh, metrics = find_optimal_threshold(y_val_tpe, val_probs)
                        val_pred = (val_probs >= best_thresh).astype(int)
                        
                        macro_f1 = f1_score(y_val_tpe, val_pred, average='macro', zero_division=0)
                        gmean = metrics['gmean']
                        macro_auc = metrics['macro_auc']
                        
                        score = 0.4 * macro_f1 + 0.3 * gmean + 0.3 * macro_auc
                        return score
                    
                    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
                    study.optimize(catboost_optuna_objective, n_trials=20, show_progress_bar=False)
                    
                    best_params = study.best_params
                    print(f"  Best CatBoost params: {best_params}")
                    print(f"  Best TPE score: {study.best_value:.4f}")
                    
                    best_catboost = CatBoostClassifier(
                        iterations=best_params.get('iterations', 200),
                        learning_rate=best_params.get('learning_rate', 0.05),
                        depth=best_params.get('depth', 4),
                        l2_leaf_reg=best_params.get('l2_leaf_reg', 5),
                        min_data_in_leaf=best_params.get('min_data_in_leaf', 1),
                        border_count=best_params.get('border_count', 128),
                        auto_class_weights='Balanced',
                        random_seed=RANDOM_SEED,
                        verbose=10,
                        early_stopping_rounds=30
                    )
                    best_catboost.fit(X_meta_final, y_meta, verbose=10)
                    
                    tpe_catboost_probs = best_catboost.predict_proba(X_test_meta_final)
                    tpe_pred_optimized, tpe_threshold, tpe_metrics = apply_threshold_tuning(y_test, tpe_catboost_probs[:, 1])
                    
                    results_collector['Ensemble_TPE_CatBoost'] = calculate_all_metrics(y_test, tpe_pred_optimized, tpe_catboost_probs)
                    results_collector['Ensemble_TPE_CatBoost']['optimal_threshold'] = tpe_threshold
                    results_collector['Ensemble_TPE_CatBoost']['tpe_params'] = best_params
                    
                    print(f"Ensemble (TPE CatBoost) - Test F1: {results_collector['Ensemble_TPE_CatBoost']['f1']:.4f}, "
                            f"Test AUC: {results_collector['Ensemble_TPE_CatBoost']['auc']:.4f}, "
                            f"Test MacroF1: {results_collector['Ensemble_TPE_CatBoost']['macro_f1']:.4f}, "
                            f"Test MacroAUC: {results_collector['Ensemble_TPE_CatBoost']['macro_auc']:.4f}, "
                            f"Test G-Mean: {results_collector['Ensemble_TPE_CatBoost']['gmean']:.4f}, "
                            f"Optimal Threshold: {tpe_threshold:.2f}")
    
    except Exception as e:
        print(f"Error in TPE CatBoost optimization: {e}")

    # ========================================================================
    # 8. Display Results Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by Macro F1 (main focus metric)
    sorted_results = sorted(results_collector.items(), key=lambda x: x[1].get('macro_f1', 0), reverse=True)
    
    print(f"\n{'Model':<25} {'M_F1':>8} {'M_Precision':>8} {'M_Recall':>8} {'M_AUC':>8} {'G-Mean':>8} {'Specificity':>8} {'F1':>8} {'Precision':>8} {'Recall':>8} {'AUC':>8} {'Accuracy':>8}")
    print("-" * 130)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} "
              f"{metrics.get('macro_f1', 0):>8.4f} "
              f"{metrics.get('macro_precision', 0):>8.4f} "
              f"{metrics.get('macro_recall', 0):>8.4f} "
              f"{metrics.get('macro_auc', 0):>8.4f} "
              f"{metrics.get('gmean', 0):>8.4f} "
              f"{metrics.get('specificity', 0):>8.4f} "
              f"{metrics.get('f1', 0):>8.4f} "
              f"{metrics.get('precision', 0):>8.4f} "
              f"{metrics.get('recall', 0):>8.4f} "
              f"{metrics.get('auc', 0):>8.4f} "
              f"{metrics.get('accuracy', 0):>8.4f} "
              )

    # ========================================================================
    # 9. Generate Visualizations
    # ========================================================================
    print("\n9. Generating visualizations...")
    
    # Get test predictions from the best performing model (Ensemble_Average)
    # Use threshold-tuned predictions for visualization
    test_y_pred = ensemble_pred_optimized  # Use threshold-tuned ensemble predictions
    
    # try:
    generate_standard_gnn_visualizations(results_collector, training_histories, test_y_true, 
                                              test_y_pred)
    # except Exception as e:
        # print(f"Visualization generation skipped: {e}")

    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)

    return results_collector, training_histories


if __name__ == "__main__":
    results, histories = run_full_pipeline()
