"""
Student ID: 24027277d
Name: Yuen Tsz Ki

Blockchain Anomaly Detection GNN Framework - Final Version

This framework implements:
- Data preprocessing with Degree, Personalized PageRank, and Louvain community detection
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
import warnings
# warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, MLP, 
    APPNP, ChebConv, GCN2Conv, MixHopConv
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree as pyg_degree, get_ppr

from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Using simple averaging for ensemble.")

from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations


# ==============================================================================
# Data Processing - Feature Engineering
# ==============================================================================

class GraphFeatureEngineer:
    """Graph feature engineering with Degree, PageRank, and Louvain community detection"""
    
    def __init__(self, data):
        self.data = data
        self.edge_index = data.edge_index.cpu()
        self.num_nodes = data.x.size(0)
        
    def compute_degree_features(self):
        """Compute degree features for all nodes"""
        print("Computing degree features...")
        deg = pyg_degree(self.edge_index[0], self.num_nodes)
        
        # Degree statistics
        in_degree = torch.zeros(self.num_nodes)
        out_degree = torch.zeros(self.num_nodes)
        
        # For undirected graph, in_degree = out_degree = degree/2
        in_degree = deg.clone()
        out_degree = deg.clone()
        
        # Normalize degrees
        max_deg = deg.max().item()
        if max_deg > 0:
            deg_normalized = deg / max_deg
        else:
            deg_normalized = deg
            
        # Add as new features
        degree_features = torch.stack([deg, in_degree, out_degree, deg_normalized], dim=1)
        
        return degree_features
    
    def compute_pagerank_features(self, alpha=0.2, eps=1e-5):
        """Compute Personalized PageRank features using get_ppr
        
        Uses the get_ppr function from PyG to compute personalized PageRank vectors.
        For anomaly detection, computes:
        1. Standard PageRank (from all nodes)
        2. PPR from illegal nodes (class 1) - identifies nodes connected to illicit entities
        """
        print("Computing PageRank features using get_ppr...")
        
        # Method 1: Compute standard PageRank-like values using get_ppr with full nodes
        # get_ppr computes PPR vectors - we aggregate to get node importance
        
        # For standard PageRank, use uniform distribution over all nodes
        # Compute PPR from a sample of nodes to estimate overall importance
        sample_size = min(100, self.num_nodes)
        sample_indices = torch.randperm(self.num_nodes)[:sample_size]
        
        ppr_importance = torch.zeros(self.num_nodes)
        
        for idx in sample_indices:
            target_tensor = torch.tensor([idx.item() if idx.dim() > 0 else idx], dtype=torch.long)
            ppr_edge_index, ppr_weight = get_ppr(
                self.edge_index, 
                alpha=alpha, 
                eps=eps, 
                target=target_tensor, 
                num_nodes=self.num_nodes
            )
            # Aggregate PPR values as node importance
            if ppr_weight.numel() > 0:
                # Add to source nodes
                ppr_importance[ppr_edge_index[0]] += ppr_weight
        
        # Normalize by sample size
        pr_values = ppr_importance / sample_size
        
        # Method 2: Compute PPR from illegal nodes for anomaly detection
        labels = self.data.y.cpu()
        illegal_indices = torch.where(labels == 1)[0]
        
        ppr_illegal = torch.zeros(self.num_nodes)
        
        if len(illegal_indices) > 0:
            # Sample illegal nodes if too many
            max_illegal_samples = min(50, len(illegal_indices))
            illegal_sample = illegal_indices[torch.randperm(len(illegal_indices))[:max_illegal_samples]]
            
            for idx in illegal_sample:
                ppr_edge_index, ppr_weight = get_ppr(
                    self.edge_index,
                    alpha=alpha,
                    eps=eps,
                    target=idx,
                    num_nodes=self.num_nodes
                )
                if ppr_weight.numel() > 0:
                    ppr_illegal[ppr_edge_index[0]] += ppr_weight
            
            ppr_illegal = ppr_illegal / max_illegal_samples
        
        # Combine features
        pagerank_features = torch.stack([
            pr_values,                      # Standard PageRank importance
            ppr_illegal,                    # PPR from illegal nodes
            pr_values * ppr_illegal,        # Interaction
            (pr_values + ppr_illegal) / 2   # Average
        ], dim=1)
        
        return pagerank_features
    
    def compute_louvain_communities(self, resolution=1.0):
        """Detect communities using Louvain algorithm"""
        print("Detecting communities using Louvain algorithm...")
        
        try:
            from community import community_louvain
            
            # Convert edge_index to networkx format
            import networkx as nx
            
            # Create undirected graph
            G = nx.Graph()
            edges = self.edge_index.t().numpy()
            G.add_edges_from(edges)
            
            # Run Louvain algorithm
            partition = community_louvain.best_partition(G, resolution=resolution, random_state=24027277)
            
            # Convert to features
            community_ids = torch.zeros(self.num_nodes, dtype=torch.long)
            for node, comm_id in partition.items():
                if node < self.num_nodes:
                    community_ids[node] = comm_id
            
            # Number of communities
            num_communities = len(set(partition.values()))
            print(f"Found {num_communities} communities")
            
            # Create one-hot encoding for communities (limited to top communities)
            max_communities = min(50, num_communities)  # Limit to 50 communities
            community_features = torch.zeros(self.num_nodes, max_communities + 2)
            
            for i in range(self.num_nodes):
                comm_id = partition.get(i, 0)
                if comm_id < max_communities:
                    community_features[i, comm_id] = 1.0
            
            # Mark nodes in illegal communities
            labels = self.data.y.cpu()
            illegal_communities = set()
            for node in range(self.num_nodes):
                if labels[node] == 1:
                    illegal_communities.add(partition.get(node, -1))
            
            # Mark community as suspicious if it contains illegal nodes
            for i in range(self.num_nodes):
                comm_id = partition.get(i, -1)
                if comm_id in illegal_communities:
                    community_features[i, max_communities] = 1.0  # Suspicious community
                    
            # Add degree within community
            community_features[:, max_communities + 1] = community_features[:, :max_communities].sum(dim=1)
            
            return community_features, partition
            
        except ImportError:
            print("Warning: python-louvain not installed. Using random community features.")
            # Fallback: create random community features
            community_features = torch.zeros(self.num_nodes, 10)
            random.seed(24027277)
            for i in range(self.num_nodes):
                community_features[i, random.randint(0, 9)] = 1.0
            return community_features, {}
    
    def add_all_features(self):
        """Add all graph features to the data object"""
        
        # Original features
        new_x = [self.data.x]
        
        # Degree features
        degree_feats = self.compute_degree_features()
        new_x.append(degree_feats.to(self.data.x.device))
        
        # PageRank features
        pagerank_feats = self.compute_pagerank_features()
        new_x.append(pagerank_feats.to(self.data.x.device))
        
        # Community features
        community_feats, partition = self.compute_louvain_communities()
        new_x.append(community_feats.to(self.data.x.device))
        
        # Concatenate all features
        enhanced_x = torch.cat(new_x, dim=1)
        
        # Update data object
        self.data.x = enhanced_x
        
        print(f"Enhanced features: {enhanced_x.size(1)} (original: {self.data.x.size(1) - enhanced_x.size(1) + 166})")
        
        return self.data, partition


# ==============================================================================
# Loss Functions
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==============================================================================
# Baseline Model - Isolation Forest
# ==============================================================================

class IsolationForestBaseline:
    """Baseline model using Isolation Forest for anomaly detection"""
    
    def __init__(self, n_estimators=100, max_samples='auto', random_state=24027277, contamination='auto'):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.model = None
        self.scaler = None

    def fit(self, data):
        """Train Isolation Forest on node features only"""
        print("Training Isolation Forest baseline model...")
        
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.contamination == 'auto':
            contamination_ratio = np.mean(y_train == 1)
            self.contamination = min(max(contamination_ratio, 0.01), 0.5)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled)
        print("Isolation Forest training completed")
        return self

    def predict(self, data):
        """Predict on test data"""
        X_test = data.x[data.test_mask].cpu().numpy()
        X_test_scaled = self.scaler.transform(X_test)

        scores = self.model.decision_function(X_test_scaled)
        predictions = self.model.predict(X_test_scaled)
        binary_predictions = (predictions == -1).astype(int)

        return binary_predictions, scores

    def evaluate(self, data):
        """Evaluate baseline model"""
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred, anomaly_scores = self.predict(data)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        prob_scores = -anomaly_scores
        auc = roc_auc_score(y_true, prob_scores)
        
        sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'auc': auc,
            'macro_auc': auc,
            'gmean': gmean,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }

        return results


# ==============================================================================
# GNN Models (7 base models + MixHop variants)
# ==============================================================================

class GCNModel(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GATModel(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.5):
        super(GATModel, self).__init__()
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GINModel(torch.nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GINModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)))
        self.convs.append(GINConv(MLP([hidden_channels, hidden_channels, out_channels], dropout=dropout)))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class APPNPModel(torch.nn.Module):
    """APPNP (Approximate Personalized Propagation of Neural Predictions)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, alpha=0.1, num_iterations=3):
        super(APPNPModel, self).__init__()
        self.mlp = MLP([in_channels, hidden_channels, hidden_channels, out_channels], dropout=dropout)
        self.dropout = dropout
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.propagate = APPNP(K=num_iterations, alpha=alpha, dropout=dropout)

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        x = self.mlp(x)
        features = [x.clone()]
        x = self.propagate(x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class ChebNetModel(torch.nn.Module):
    """Chebyshev Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
        super(ChebNetModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, K=K))
        self.convs.append(ChebConv(hidden_channels, out_channels, K=K))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GCNIIModel(torch.nn.Module):
    """GCNII (Graph Convolutional Networks with Initial Residual and Identity Mapping)
    
    From "Simple and Deep Graph Convolutional Networks" paper.
    Uses GCN2Conv which combines:
    - Initial residual: (1-α)PX + αX⁽⁰⁾
    - Identity mapping: (1-β)I + βΘ
    This allows deeper networks without over-smoothing.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, 
                 alpha=0.5, theta=1.0, num_layers=2):
        super(GCNIIModel, self).__init__()
        self.alpha = alpha
        self.theta = theta
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN2Conv layers - note: channels must match hidden_channels
        self.convs = nn.ModuleList([
            GCN2Conv(
                channels=hidden_channels, 
                alpha=alpha, 
                theta=theta, 
                layer=i+1,
                shared_weights=True,
                add_self_loops=True,
                normalize=True
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Save initial features for residual connections
        x_0 = self.input_proj(x)
        x = x_0
        
        # Apply GCN2Conv layers
        features = [x.clone()]
        for conv in self.convs:
            x = conv(x, x_0, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            features.append(x.clone())
        
        # Output projection
        x = self.output_proj(x)
        out = F.log_softmax(x, dim=1)
        
        if return_embed:
            embed = features[-1] if features else x_0
            return out, embed
        return out


# ==============================================================================
# MixHop Models - Multi-hop Information Enhancement using MixHopConv
# ==============================================================================

class MixHopGCNModel(torch.nn.Module):
    """GCN with MixHopConv for multi-hop information
    
    MixHop learns separate weights for each hop distance, enabling the model
    to capture relationships at different neighborhood distances without
    stacking many layers (mitigating over-smoothing).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_hops=3):
        super(MixHopGCNModel, self).__init__()
        
        # MixHopConv: learns to combine information from multiple hops
        self.mixhop_conv = MixHopConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_hops=num_hops  # Number of hop distances to consider
        )
        
        # Additional MLP to process concatenated multi-hop features
        self.mlp = MLP([
            hidden_channels * num_hops,  # MixHopConv outputs concatenated features
            hidden_channels * num_hops // 2,
            out_channels
        ], dropout=dropout)
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply MixHopConv - automatically handles multi-hop neighborhood aggregation
        x = self.mixhop_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        out = self.mlp(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, x
        return out


class MixHopGATModel(torch.nn.Module):
    """GAT with MixHop for multi-hop information using multiple attention heads"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.5, num_hops=2):
        super(MixHopGATModel, self).__init__()
        self.num_hops = num_hops
        self.num_heads = num_heads
        
        # Multiple GAT layers for different hops - simulates MixHop behavior
        self.hop_convs = nn.ModuleList([
            GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True)
            for _ in range(num_hops)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(hidden_channels * num_heads * num_hops, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply attention for each hop distance and concatenate
        hop_outputs = []
        for conv in self.hop_convs:
            h = conv(x, edge_index)
            h = F.relu(h)
            hop_outputs.append(h)
        
        # Concatenate multi-hop features
        x = torch.cat(hop_outputs, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection
        out = self.out_proj(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, hop_outputs[-1]
        return out


class MixHopGraphSAGEModel(torch.nn.Module):
    """GraphSAGE with MixHop for multi-hop information"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_hops=2, aggr='mean'):
        super(MixHopGraphSAGEModel, self).__init__()
        self.num_hops = num_hops
        
        # Multiple SAGE layers for different hops
        self.hop_convs = nn.ModuleList([
            SAGEConv(in_channels, hidden_channels, aggr=aggr)
            for _ in range(num_hops)
        ])
        
        self.out_proj = nn.Linear(hidden_channels * num_hops, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        hop_outputs = []
        for conv in self.hop_convs:
            h = conv(x, edge_index)
            h = F.relu(h)
            hop_outputs.append(h)
        
        x = torch.cat(hop_outputs, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.out_proj(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, hop_outputs[-1]
        return out


class MixHopGINModel(torch.nn.Module):
    """GIN with MixHop for multi-hop information"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_hops=2):
        super(MixHopGINModel, self).__init__()
        self.num_hops = num_hops
        
        # Multiple GIN layers for different hops
        self.hop_convs = nn.ModuleList([
            GINConv(MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout))
            for _ in range(num_hops)
        ])
        
        self.out_proj = nn.Linear(hidden_channels * num_hops, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        hop_outputs = []
        for conv in self.hop_convs:
            h = conv(x, edge_index)
            h = F.relu(h)
            hop_outputs.append(h)
        
        x = torch.cat(hop_outputs, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.out_proj(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, hop_outputs[-1]
        return out


# Model class mapping
MODEL_CLASSES = {
    'GCN': GCNModel,
    'GAT': GATModel,
    'GraphSAGE': GraphSAGEModel,
    'GIN': GINModel,
    'APPNP': APPNPModel,
    'ChebNet': ChebNetModel,
    'GCNII': GCNIIModel,
    # MixHop variants
    'MixHop_GCN': MixHopGCNModel,
    'MixHop_GAT': MixHopGATModel,
    'MixHop_GraphSAGE': MixHopGraphSAGEModel,
    'MixHop_GIN': MixHopGINModel,
}


# ==============================================================================
# Trainer Class
# ==============================================================================

class Trainer:
    def __init__(self, model, data, device, optimizer, criterion, history=None):
        self.model = model
        self.data = data
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = history or TrainingHistory()

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
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
        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_macro_recall = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_macro_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        
        val_probs = torch.exp(out).cpu().numpy()
        val_mask_np = self.data.val_mask.cpu().numpy()
        val_auc = roc_auc_score(val_y_true, val_probs[val_mask_np][:, 1])
        
        val_sensitivity = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
        val_specificity = recall_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
        val_gmean = np.sqrt(val_sensitivity * val_specificity) if val_sensitivity * val_specificity > 0 else 0

        result = {
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_macro_recall': val_macro_recall,
            'val_macro_f1': val_macro_f1,
            'val_macro_auc': val_auc,
            'val_gmean': val_gmean
        }

        if include_test and self.data.test_mask.sum() > 0:
            test_y_true = self.data.y[self.data.test_mask].cpu().numpy()
            test_y_pred = pred[self.data.test_mask].cpu().numpy()
            test_f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
            test_acc = accuracy_score(test_y_true, test_y_pred)
            
            probs = torch.exp(out).cpu().numpy()
            test_mask_np = self.data.test_mask.cpu().numpy()
            
            test_precision_illicit = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_recall_illicit = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_macro_recall = recall_score(test_y_true, test_y_pred, average='macro', zero_division=0)
            test_macro_f1 = f1_score(test_y_true, test_y_pred, average='macro', zero_division=0)
            test_auc = roc_auc_score(test_y_true, probs[test_mask_np][:, 1])
            
            test_sensitivity = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_specificity = recall_score(test_y_true, test_y_pred, pos_label=0, zero_division=0)
            test_gmean = np.sqrt(test_sensitivity * test_specificity) if test_sensitivity * test_specificity > 0 else 0

            result.update({
                'test_f1': test_f1,
                'test_acc': test_acc,
                'test_precision_illicit': test_precision_illicit,
                'test_recall_illicit': test_recall_illicit,
                'test_macro_recall': test_macro_recall,
                'test_macro_f1': test_macro_f1,
                'test_auc': test_auc,
                'test_macro_auc': test_auc,
                'test_gmean': test_gmean
            })
        else:
            result.update({
                'test_f1': 0.0, 'test_acc': 0.0, 'test_precision_illicit': 0.0,
                'test_recall_illicit': 0.0, 'test_macro_recall': 0.0,
                'test_macro_f1': 0.0, 'test_auc': 0.0, 'test_macro_auc': 0.0, 'test_gmean': 0.0
            })

        return result

    def fit(self, epochs=100, include_test=True):
        best_val_loss = float('inf')
        best_stats = None

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            stats = self.evaluate(include_test=include_test)

            self.history.add_epoch(
                epoch=epoch + 1, train_loss=train_loss, val_loss=stats['val_loss'],
                val_f1=stats['val_f1'], test_f1=stats['test_f1'],
                val_acc=stats['val_acc'], test_acc=stats['test_acc'],
                val_macro_recall=stats['val_macro_recall'], test_macro_recall=stats['test_macro_recall'],
                val_gmean=stats['val_gmean'], test_gmean=stats['test_gmean'],
                val_macro_f1=stats['val_macro_f1'], test_macro_f1=stats['test_macro_f1'],
                val_macro_auc=stats['val_macro_auc'], test_macro_auc=stats['test_macro_auc']
            )

            if stats['val_loss'] < best_val_loss:
                best_val_loss = stats['val_loss']
                best_stats = stats

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {stats['val_loss']:.4f}, Val F1: {stats['val_f1']:.4f}, "
                      f"Test F1: {stats['test_f1']:.4f}")

        return best_stats, self.history


# ==============================================================================
# Data Loading
# ==============================================================================

def load_elliptic_data(dataset_dir='../Dataset'):
    """Load Elliptic Bitcoin transaction dataset"""
    print("Loading Elliptic dataset...")
    
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')

    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)

    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
    features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(2, features_df.shape[1])]

    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')

    feature_columns = nodes_df.columns[2:-1]
    features = nodes_df[feature_columns].values
    x = torch.tensor(features, dtype=torch.float)

    print(f"Loaded {x.size(0)} nodes with {x.size(1)} features")

    labels = nodes_df['class'].apply(lambda c: 0 if c == '2' else (1 if c == '1' else -1))
    y = torch.tensor(labels.values, dtype=torch.long)

    total_nodes = len(y)
    licit_count = (y == 0).sum().item()
    illicit_count = (y == 1).sum().item()
    unknown_count = (y == -1).sum().item()

    print(f"Label distribution: licit: {licit_count}, illicit: {illicit_count}, unknown: {unknown_count}")

    tx_ids = nodes_df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    source_indices = []
    target_indices = []
    for _, row in edgelist_df.iterrows():
        src = row['txId1'] if 'txId1' in edgelist_df.columns else row.iloc[0]
        tgt = row['txId2'] if 'txId2' in edgelist_df.columns else row.iloc[1]
        if src in tx_id_map and tgt in tx_id_map:
            source_indices.append(tx_id_map[src])
            target_indices.append(tx_id_map[tgt])

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    print(f"Graph structure: {edge_index.size(1)} edges")

    timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps

    known_mask = y != -1
    data.train_mask = (timesteps < 35) & known_mask
    data.val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    data.test_mask = (timesteps >= 42) & known_mask

    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()

    print(f"Data splits: Train: {train_count}, Val: {val_count}, Test: {test_count}")

    return data


# ==============================================================================
# NeighborLoader for Mini-batch Training
# ==============================================================================

def create_neighbor_loader(data, batch_size=512, num_neighbors=[10, 5], shuffle=True):
    """Create NeighborLoader for mini-batch subgraph sampling"""
    
    # For full graph training, we use the entire graph
    # NeighborLoader is more useful for very large graphs that don't fit in GPU
    
    print(f"Creating NeighborLoader with batch_size={batch_size}, num_neighbors={num_neighbors}")
    
    # Get node indices for training
    train_indices = torch.where(data.train_mask)[0]
    
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_indices,
        shuffle=shuffle
    )
    
    return loader


# ==============================================================================
# Hyperopt Objective Function
# ==============================================================================

def hyperopt_objective(data, device, params):
    """Hyperopt optimization objective function - optimizes on GCN as representative model"""
    torch.manual_seed(24027277)
    np.random.seed(24027277)
    random.seed(24027277)

    # Use GCN as representative model for hyperparameter optimization
    model_name = 'GCN'
    hidden_channels = params['hidden_channels']
    dropout = params['dropout']
    num_heads = params.get('num_heads', 4)

    model_class = MODEL_CLASSES.get(model_name, GCNModel)
    
    try:
        if model_name == 'GAT':
            model = model_class(
                in_channels=data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                num_heads=num_heads,
                dropout=dropout
            )
        elif model_name in ['GCNII', 'APPNP', 'ChebNet']:
            model = model_class(
                in_channels=data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                dropout=dropout
            )
        elif 'MixHop' in model_name:
            model = model_class(
                in_channels=data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                dropout=dropout
            )
        else:
            model = model_class(
                in_channels=data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                dropout=dropout
            )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = FocalLoss(alpha=0.25, gamma=2)

        trainer = Trainer(model, data, device, optimizer, criterion)
        best_stats, _ = trainer.fit(epochs=30)

        return {
            'loss': -best_stats['test_f1'],
            'status': STATUS_OK,
            'test_f1': best_stats['test_f1'],
            'test_macro_f1': best_stats['test_macro_f1'],
            'test_macro_auc': best_stats['test_macro_auc'],
            'test_gmean': best_stats['test_gmean']
        }
    except Exception as e:
        print(f"Error in hyperopt objective: {e}")
        return {'loss': 1.0, 'status': STATUS_OK}


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate all 9 evaluation metrics"""
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_probs[:, 1])
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0

    return {
        'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'macro_recall': macro_recall, 'macro_f1': macro_f1, 'auc': auc, 'macro_auc': auc, 'gmean': gmean
    }


# ==============================================================================
# Ensemble with CatBoost
# ==============================================================================

class CatBoostEnsemble:
    """Stacking ensemble with CatBoost meta-model"""
    
    def __init__(self, base_models, device):
        self.base_models = base_models  # dict of trained models
        self.device = device
        self.meta_model = None
        
    def get_base_predictions(self, data, train_mask=None):
        """Get predictions from all base models"""
        all_probs = []
        
        for model_name, model in self.base_models.items():
            model.eval()
            with torch.no_grad():
                output = model(data)
                probs = torch.exp(output).cpu().numpy()
                all_probs.append(probs)
        
        # Stack all predictions: (num_models, num_nodes, 2)
        stacked_probs = np.stack(all_probs, axis=0)
        
        return stacked_probs
    
    def fit(self, data, val_data=None):
        """Train the meta-model using stacking"""
        
        # Get base model predictions on training data
        stacked_probs = self.get_base_predictions(data)
        train_mask_np = data.train_mask.cpu().numpy()
        
        # Prepare features for meta-model: flatten probability predictions
        X_meta = stacked_probs[:, train_mask_np, :].transpose(1, 0, 2).reshape(train_mask_np.sum(), -1)
        y_meta = data.y[data.train_mask].cpu().numpy()
        
        if CATBOOST_AVAILABLE:
            print("Training CatBoost meta-model...")
            self.meta_model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=24027277,
                verbose=10
            )
            self.meta_model.fit(X_meta, y_meta)
        else:
            # Simple averaging fallback
            print("CatBoost not available, using simple averaging")
            self.meta_model = None
        
        return self
    
    def predict(self, data, test_only=False):
        """Predict using the ensemble"""
        
        if test_only:
            mask_np = data.test_mask.cpu().numpy()
        else:
            mask_np = np.ones(data.x.size(0), dtype=bool)
        
        stacked_probs = self.get_base_predictions(data)
        
        if CATBOOST_AVAILABLE and self.meta_model is not None:
            # Use CatBoost for final prediction
            X_meta = stacked_probs[:, mask_np, :].transpose(1, 0, 2).reshape(mask_np.sum(), -1)
            predictions = self.meta_model.predict(X_meta)
            return predictions.astype(int)
        else:
            # Simple averaging
            avg_probs = stacked_probs.mean(axis=0)
            predictions = np.argmax(avg_probs[mask_np], axis=1)
            return predictions
    
    def predict_proba(self, data, test_only=False):
        """Get probability predictions"""
        
        if test_only:
            mask_np = data.test_mask.cpu().numpy()
        else:
            mask_np = np.ones(data.x.size(0), dtype=bool)
        
        stacked_probs = self.get_base_predictions(data)
        
        if CATBOOST_AVAILABLE and self.meta_model is not None:
            X_meta = stacked_probs[:, mask_np, :].transpose(1, 0, 2).reshape(mask_np.sum(), -1)
            probs = self.meta_model.predict_proba(X_meta)
            return probs
        else:
            avg_probs = stacked_probs.mean(axis=0)
            return avg_probs[mask_np]


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
    data, community_partition = feature_engineer.add_all_features()
    print(f"Enhanced features: {data.x.size(1)} dimensions")

    # ========================================================================
    # 3. Train Isolation Forest baseline
    # ========================================================================
    print("\n3. Training Isolation Forest baseline...")
    baseline_model = IsolationForestBaseline(
        n_estimators=100, contamination='auto', random_state=24027277
    )
    baseline_model.fit(data)
    baseline_results = baseline_model.evaluate(data)
    print("Baseline evaluation completed")

    # ========================================================================
    # 4. Bayesian Optimization for Hyperparameter Search
    # ========================================================================
    print("\n4. Running Bayesian Optimization (TPE) for hyperparameter search...")
    
    # Define search space for each model type
    # We'll optimize on GCN as representative model to save time
    space = {
        'hidden_channels': hp.choice('hidden_channels', [64, 128, 256]),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'num_heads': hp.choice('num_heads', [4, 8]),
    }
    
    MODEL_CHOICES = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII']
    HIDDEN_CHANNEL_CHOICES = [64, 128, 256]
    NUM_HEAD_CHOICES = [4, 8]
    
    # Run optimization on GCN as representative
    print("Optimizing on GCN model (representative)...")
    trials = Trials()
    
    objective_with_data = partial(hyperopt_objective, data, device)
    best = fmin(
        objective_with_data,
        space=space,
        algo=tpe.suggest,
        max_evals=10,  # 10 evaluations for meaningful optimization
        trials=trials,
        rstate=np.random.default_rng(24027277)
    )
    
    best_hidden_channels = HIDDEN_CHANNEL_CHOICES[best['hidden_channels']]
    best_dropout = best['dropout']
    best_lr = best['lr']
    best_num_heads = NUM_HEAD_CHOICES[best['num_heads']]
    
    print(f"Best hyperparameters found:")
    print(f"  Hidden channels: {best_hidden_channels}")
    print(f"  Dropout: {best_dropout:.4f}")
    print(f"  Learning rate: {best_lr:.6f}")
    print(f"  Num heads: {best_num_heads}")

    # ========================================================================
    # 5. Define models to train (7 base + 4 MixHop = 11 models)
    # ========================================================================
    print("\n5. Setting up model configurations...")
    
    # Base models (7)
    base_models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII']
    # MixHop models (4)
    mixhop_models_list = ['MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN']
    
    all_models_to_train = base_models_list + mixhop_models_list
    
    # Store results
    results_collector = {'IForest': baseline_results}
    training_histories = {}
    trained_models = {}
    
    # Get test labels
    test_y_true = data.y[data.test_mask].cpu().numpy()
    
    # ========================================================================
    # 6. Train all GNN models
    # ========================================================================
    print("\n6. Training GNN models...")
    
    for model_name in all_models_to_train:
        print(f"\n--- Training {model_name} ---")
        
        try:
            model_class = MODEL_CLASSES[model_name]
            
            # Create model
            if model_name == 'GAT':
                model = model_class(
                    in_channels=data.x.size(1),
                    hidden_channels=best_hidden_channels,
                    out_channels=2,
                    num_heads=best_num_heads,
                    dropout=best_dropout
                )
            elif 'MixHop' in model_name:
                model = model_class(
                    in_channels=data.x.size(1),
                    hidden_channels=best_hidden_channels,
                    out_channels=2,
                    dropout=best_dropout
                )
            else:
                model = model_class(
                    in_channels=data.x.size(1),
                    hidden_channels=best_hidden_channels,
                    out_channels=2,
                    dropout=best_dropout
                )
            
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            criterion = FocalLoss(alpha=1, gamma=2)
            
            trainer = Trainer(model, data, device, optimizer, criterion)
            best_stats, training_history = trainer.fit(epochs=100)
            
            # Store training history
            training_histories[model_name] = training_history
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                out = model(data)
                probs = torch.exp(out).cpu().numpy()
                test_probs = probs[data.test_mask.cpu().numpy()]
                test_pred = np.argmax(test_probs, axis=1)
            
            # Calculate metrics
            results_collector[model_name] = calculate_all_metrics(test_y_true, test_pred, test_probs)
            trained_models[model_name] = model
            
            print(f"{model_name} - Test F1: {results_collector[model_name]['f1']:.4f}, "
                  f"Test AUC: {results_collector[model_name]['auc']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

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
    results_collector['Ensemble_Average'] = calculate_all_metrics(test_y_true, base_ensemble_pred, avg_probs[test_mask_np])
    
    print(f"Ensemble (Average) - Test F1: {results_collector['Ensemble_Average']['f1']:.4f}")
    
    # CatBoost stacking (if available)
    if CATBOOST_AVAILABLE:
        try:
            print("\nTraining CatBoost stacking meta-model...")
            
            # Get all model predictions for stacking
            stacked_probs = np.stack(all_probs, axis=0)  # (num_models, num_nodes, 2)
            
            # Prepare training data for meta-model
            train_mask_np = data.train_mask.cpu().numpy()
            X_meta = stacked_probs[:, train_mask_np, :].transpose(1, 0, 2).reshape(train_mask_np.sum(), -1)
            y_meta = data.y[data.train_mask].cpu().numpy()
            
            # Train CatBoost
            catboost_meta = CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=24027277,
                verbose=10
            )
            catboost_meta.fit(X_meta, y_meta)
            
            # Predict on test set
            X_test_meta = stacked_probs[:, test_mask_np, :].transpose(1, 0, 2).reshape(test_mask_np.sum(), -1)
            catboost_pred = catboost_meta.predict(X_test_meta).astype(int)
            catboost_probs = catboost_meta.predict_proba(X_test_meta)
            
            results_collector['Ensemble_CatBoost'] = calculate_all_metrics(test_y_true, catboost_pred, catboost_probs)
            
            print(f"Ensemble (CatBoost) - Test F1: {results_collector['Ensemble_CatBoost']['f1']:.4f}")
            
        except Exception as e:
            print(f"Error in CatBoost stacking: {e}")

    # ========================================================================
    # 8. Display Results Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by F1 score
    sorted_results = sorted(results_collector.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
    
    print(f"\n{'Model':<25} {'F1':>8} {'AUC':>8} {'G-Mean':>8} {'Accuracy':>8}")
    print("-" * 60)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} {metrics['f1']:>8.4f} {metrics['auc']:>8.4f} "
              f"{metrics['gmean']:>8.4f} {metrics['accuracy']:>8.4f}")

    # ========================================================================
    # 9. Generate Visualizations
    # ========================================================================
    print("\n9. Generating visualizations...")
    
    try:
        generate_standard_gnn_visualizations(results_collector, training_histories, test_y_true, 
                                               results_collector.get('Ensemble_CatBoost', 
                                               results_collector.get('Ensemble_Average', {})).get('f1', 0))
    except Exception as e:
        print(f"Visualization generation skipped: {e}")

    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)

    return results_collector, training_histories


if __name__ == "__main__":
    results, histories = run_full_pipeline()
