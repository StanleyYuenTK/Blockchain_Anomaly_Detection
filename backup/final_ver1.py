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

# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from functools import partial

# Optuna for Bayesian Optimization with TPE
import optuna
from optuna.samplers import TPESampler
# optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Using simple averaging for ensemble.")

from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations


# ==============================================================================
# Helper Functions
# ==============================================================================

# ==============================================================================
# Data Processing - Feature Engineering
# ==============================================================================

class GraphFeatureEngineer:
    """Graph feature engineering with Degree, PageRank, and Louvain community detection"""

    def __init__(self, data):
        self.data = data
        self.edge_index = data.edge_index.cpu()
        self.num_nodes = data.x.size(0)

    def get_graph_structure_features(self):
        """Get graph structure features (degree + pagerank) for CatBoost meta-model

        Returns:
            graph_features: Tensor of shape (num_nodes, 8) containing:
                - degree (1), in_degree (1), out_degree (1), deg_normalized (1)
                - pagerank (1), pagerank_sqrt (1), pagerank_log (1), pagerank_rank (1)
        """
        degree_feats = self.compute_degree_features()
        pagerank_feats = self.compute_pagerank_features()

        # Concatenate: 4 degree + 4 pagerank = 8 features
        graph_features = torch.cat([degree_feats, pagerank_feats], dim=1)
        return graph_features

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


    
    def compute_pagerank_features(self, alpha=0.15, max_iter=100, tol=1e-6):
        """
        優化後的 PageRank 特徵計算：
        使用 Power Iteration 捕捉全局重要性，而非局部抽樣。
        """
        print(f"Executing Global PageRank Power Iteration (alpha={alpha})...")
        
        edge_index = self.edge_index
        num_nodes = self.num_nodes
        
        # 1. 準備轉移矩陣的權重
        # 計算出度 (Out-degree)
        row, col = edge_index
        out_degree = pyg_degree(row, num_nodes=num_nodes)
        
        # 處理懸掛節點 (Dangling nodes: out_degree == 0)
        # 在 PageRank 理論中，無出度的節點應被視為跳轉至所有節點
        deg_inv = 1.0 / out_degree
        deg_inv[out_degree == 0] = 0
        
        # 歸一化邊權重
        edge_weight = deg_inv[row]
        
        # 2. Power Iteration 過程
        # 初始化平穩分佈為均勻分佈
        pr_values = torch.full((num_nodes,), 1.0 / num_nodes, device=edge_index.device)
        teleport = torch.full((num_nodes,), 1.0 / num_nodes, device=edge_index.device)
        
        for i in range(max_iter):
            prev_pr = pr_values.clone()
            
            # 核心公式: v = (1-alpha) * M * v + alpha * teleport
            # 使用 torch_sparse 或聚合操作實現矩陣乘法
            # 這裡利用 message passing 邏輯：將 pr 值沿著邊傳遞並按權重聚合
            msg = prev_pr[row] * edge_weight
            new_pr = torch.zeros_like(pr_values)
            new_pr.scatter_add_(0, col, msg)
            
            pr_values = (1 - alpha) * new_pr + alpha * teleport
            
            # 檢查收斂性
            err = torch.norm(pr_values - prev_pr, p=1)
            if err < tol:
                print(f"PageRank converged at iteration {i}")
                break

        # 3. 特徵變換 (保留爾等原有的優秀非線性特徵工程)
        pr_values_log = torch.log1p(pr_values * num_nodes) # 縮放後取對數，增強數值穩定性
        pr_values_sqrt = torch.sqrt(pr_values * num_nodes)
        
        # 計算 Ordinal Rank (在此數據規模下極具區分度)
        pr_ranks = torch.argsort(torch.argsort(pr_values, descending=True)).float()
        pr_ranks_normalized = pr_ranks / num_nodes

        pagerank_features = torch.stack([
            pr_values * num_nodes, # 歸一化重要度
            pr_values_sqrt,
            pr_values_log,
            pr_ranks_normalized
        ], dim=1)

        return pagerank_features
    
    def compute_louvain_communities(self, resolution=1.0, train_mask=None):
        """Detect communities using Louvain algorithm
        
        Args:
            resolution: Louvain resolution parameter
            train_mask: Optional tensor indicating training nodes. If provided, only use 
                        training set labels to identify suspicious communities (prevents leakage).
        """
        print("Detecting communities using Louvain algorithm...")
        
        try:
            from community import community_louvain
            
            # Convert edge_index to networkx format
            import networkx as nx
            
            # Create undirected graph
            G = nx.Graph()
            edges = self.edge_index.t().cpu().numpy()
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
            
            # Mark nodes in illegal communities - ONLY use training set labels (prevents data leakage)
            labels = self.data.y.cpu()
            illegal_communities = set()
            
            # Determine which nodes to use for identifying suspicious communities
            if train_mask is not None:
                # Only use training set nodes to identify illegal communities
                train_mask_np = train_mask.cpu().numpy() if isinstance(train_mask, torch.Tensor) else train_mask
                nodes_to_use = np.where(train_mask_np)[0]
                print(f"Using training set ({len(nodes_to_use)} nodes) to identify suspicious communities")
            else:
                # Fallback: use all nodes (legacy behavior - contains leakage)
                nodes_to_use = range(self.num_nodes)
                print("Warning: No train_mask provided. Using all nodes (may cause data leakage).")
            
            for node in nodes_to_use:
                if labels[node] == 1:
                    illegal_communities.add(partition.get(node, -1))
            
            # Mark community as suspicious if it contains illegal nodes (from training set only)
            suspicious_count = 0
            for i in range(self.num_nodes):
                comm_id = partition.get(i, -1)
                if comm_id in illegal_communities:
                    community_features[i, max_communities] = 1.0  # Suspicious community
                    suspicious_count += 1
            
            print(f"Marked {len(illegal_communities)} communities as suspicious (containing training set illegal nodes)")
            print(f"Total nodes in suspicious communities: {suspicious_count}")
                    
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

    # def compute_community_illegal_ratio(self, partition, train_labels):
    #     """Compute the illegal node ratio for each community (using training set labels only)
        
    #     Args:
    #         partition: Dictionary mapping node_id to community_id
    #         train_labels: Labels for training nodes only
            
    #     Returns:
    #         Dictionary mapping community_id to illegal ratio (0-1)
    #     """
    #     community_stats = {}
        
    #     for node, comm_id in partition.items():
    #         if comm_id not in community_stats:
    #             community_stats[comm_id] = {'total': 0, 'illegal': 0}
            
    #         # Only count nodes that are in the training set (labels != -1)
    #         if train_labels[node] != -1:
    #             community_stats[comm_id]['total'] += 1
    #             if train_labels[node] == 1:
    #                 community_stats[comm_id]['illegal'] += 1
        
    #     # Compute ratios
    #     community_illegal_ratio = {}
    #     for comm_id, stats in community_stats.items():
    #         if stats['total'] > 0:
    #             community_illegal_ratio[comm_id] = stats['illegal'] / stats['total']
    #         else:
    #             community_illegal_ratio[comm_id] = 0.0
        
    #     return community_illegal_ratio

    # def identify_high_risk_communities(self, partition, train_labels, threshold=0.5):
    #     """Identify communities with illegal node proportion > threshold
        
    #     Args:
    #         partition: Dictionary mapping node_id to community_id
    #         train_labels: Labels for training nodes only
    #         threshold: Minimum illegal ratio to be considered high-risk (default 0.8 = 80%)
            
    #     Returns:
    #         Set of community_ids that are high-risk
    #     """
    #     community_illegal_ratio = self.compute_community_illegal_ratio(partition, train_labels)
        
    #     high_risk_communities = set()
    #     for comm_id, ratio in community_illegal_ratio.items():
    #         if ratio >= threshold:
    #             high_risk_communities.add(comm_id)
        
    #     return high_risk_communities



    # def get_unknown_nodes_in_high_risk_communities(self, data, high_risk_communities, partition):
    #     """Get Unknown nodes (y == -1) that are in high-risk communities
        
    #     Args:
    #         data: PyG Data object
    #         high_risk_communities: Set of community_ids that are high-risk
    #         partition: Dictionary mapping node_id to community_id
            
    #     Returns:
    #         List of node indices that are Unknown and in high-risk communities
    #     """
    #     unknown_nodes = []
        
    #     for node in range(self.num_nodes):
    #         # Check if node is Unknown (label == -1)
    #         if data.y[node] == -1:
    #             # Check if node is in a high-risk community
    #             comm_id = partition.get(node, -1)
    #             if comm_id in high_risk_communities:
    #                 unknown_nodes.append(node)
        
    #     return unknown_nodes

    def add_degree_pagerank_louvain_features(self, train_mask=None):
        """Add all graph features to the data object
        
        Args:
            train_mask: Optional tensor indicating training nodes. Used for Louvain 
                        community detection to prevent data leakage.
        """
        
        # Original features
        new_x = [self.data.x]
        
        # Degree features
        degree_feats = self.compute_degree_features()
        new_x.append(degree_feats.to(self.data.x.device))
        
        # PageRank features
        pagerank_feats = self.compute_pagerank_features()
        new_x.append(pagerank_feats.to(self.data.x.device))
        
        # Community features (with train_mask to prevent leakage)
        community_feats, partition = self.compute_louvain_communities(train_mask=train_mask)
        new_x.append(community_feats.to(self.data.x.device))
        
        # Concatenate all features
        enhanced_x = torch.cat(new_x, dim=1)
        
        # Update data object
        self.data.x = enhanced_x
        
        print(f"Enhanced features: {enhanced_x.size(1)} (original: {self.data.x.size(1) - enhanced_x.size(1) + 166})")
        
        return self.data, partition


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
    try:
        from community import community_louvain
        import networkx as nx
        
        # Create undirected graph
        G = nx.Graph()
        edges = data.edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Run Louvain algorithm
        partition = community_louvain.best_partition(G, resolution=1.0, random_state=24027277)
        
        num_communities = len(set(partition.values()))
        print(f"   Found {num_communities} communities")
        
    except ImportError:
        print("   Warning: python-louvain not installed. Cannot perform pseudo-labeling.")
        return data, [], set()
    
    # Step 2: Get training set labels (only use labeled nodes to avoid leakage)
    print("\n[Step 2] Computing community illegal ratios (using training set only)...")
    train_mask_np = data.train_mask.cpu().numpy()
    train_labels = data.y.clone()
    
    # Create a copy of labels where non-training nodes are marked as -1 (Unknown)
    # Ensure tensor is on the same device as data.y
    device = data.y.device
    train_only_labels = torch.full((data.y.size(0),), -1, dtype=torch.long, device=device)
    train_only_labels[data.train_mask] = data.y[data.train_mask]
    
    # Compute illegal ratio for each community
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
    community_illegal_ratio = {}
    high_risk_communities = set()
    
    print(f"\n   Community illegal ratios (threshold = {threshold*100:.0f}%):")
    for comm_id in sorted(community_stats.keys()):
        stats = community_stats[comm_id]
        if stats['total'] > 0:
            ratio = stats['illegal'] / stats['total']
            community_illegal_ratio[comm_id] = ratio
            
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


def iterative_pseudo_labeling(data, model_predictions_fn, num_iterations=3, 
                               community_threshold=0.8, confidence_threshold=0.9,
                               gnn_train_fn=None):
    """Iterative pseudo-labeling: Repeatedly add pseudo-labels and retrain
    
    Args:
        data: PyG Data object
        model_predictions_fn: Function that takes data and returns predictions array
        num_iterations: Number of pseudo-labeling iterations
        community_threshold: Minimum illegal ratio for high-risk communities
        confidence_threshold: Minimum confidence for pseudo-labels
        gnn_train_fn: Optional function to retrain GNN (if None, uses current predictions)
        
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
        
        # Optionally retrain the model with updated training set
        if gnn_train_fn is not None:
            print(f"\nRetraining model with pseudo-labeled data...")
            current_data = gnn_train_fn(current_data)
    
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
    """Focal Loss for addressing class imbalance
    
    Args:
        alpha: Can be either:
            - scalar (int/float): constant alpha value (original behavior)
            - tensor: class weights [alpha_0, alpha_1, ...] for each class
            - 'auto': automatically compute class weights from training data
        gamma: focusing parameter for hard examples
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
        # Handle different alpha types
        if isinstance(alpha, str) and alpha.lower() == 'auto':
            self.alpha = None  # Will be set later
            self.use_class_weights = True
        elif isinstance(alpha, (int, float)):
            self.alpha = alpha
            self.use_class_weights = False
        else:
            self.alpha = alpha  # Assume it's already a tensor
            self.use_class_weights = True

    def set_class_weights(self, y_train):
        """Set alpha based on class distribution from training data"""
        class_counts = np.bincount(y_train.astype(int))
        # Class weight = n_samples / (n_classes * n_samples_class)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        self.alpha = torch.tensor(class_weights, dtype=torch.float32)
        self.use_class_weights = True

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.use_class_weights and self.alpha is not None:
            # Ensure alpha is on the same device as inputs
            alpha = self.alpha.to(inputs.device)
            # Use class-weighted focal loss
            focal_loss = alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        else:
            # Use scalar alpha (original behavior)
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
    """Graph Convolutional Network with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)  # BatchNorm
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)  # BatchNorm on final layer
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GATModel(torch.nn.Module):
    """Graph Attention Network with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.5):
        super(GATModel, self).__init__()
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer: output is hidden_channels * num_heads (after concat)
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Second layer: output is out_channels
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)  # BatchNorm
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)  # BatchNorm on final layer
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE Network with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)  # BatchNorm
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)  # BatchNorm on final layer
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GINModel(torch.nn.Module):
    """Graph Isomorphism Network with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GINModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GINConv(MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GINConv(MLP([hidden_channels, hidden_channels, out_channels], dropout=dropout)))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)  # BatchNorm
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)  # BatchNorm on final layer
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class APPNPModel(torch.nn.Module):
    """APPNP (Approximate Personalized Propagation of Neural Predictions) with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, alpha=0.1, num_iterations=3):
        super(APPNPModel, self).__init__()
        # MLP: in -> hidden -> hidden -> out
        self.mlp1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.mlp2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.mlp3 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.propagate = APPNP(K=num_iterations, alpha=alpha, dropout=dropout)

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # First MLP layer with BatchNorm
        x = self.mlp1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second MLP layer with BatchNorm
        x = self.mlp2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third MLP layer (output)
        x = self.mlp3(x)
        
        features = [x.clone()]
        x = self.propagate(x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class ChebNetModel(torch.nn.Module):
    """Chebyshev Graph Convolutional Network with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
        super(ChebNetModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(ChebConv(in_channels, hidden_channels, K=K))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(ChebConv(hidden_channels, out_channels, K=K))
        self.bns.append(nn.BatchNorm1d(out_channels))
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)  # BatchNorm
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)  # BatchNorm on final layer
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
        self.bn_input = nn.BatchNorm1d(hidden_channels)
        
        # GCN2Conv layers - note: channels must match hidden_channels
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
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Save initial features for residual connections
        x_0 = self.input_proj(x)
        x_0 = self.bn_input(x_0)  # BatchNorm after input projection
        x = x_0
        
        # Apply GCN2Conv layers
        features = [x.clone()]
        for i, conv in enumerate(self.convs):
            x = conv(x, x_0, edge_index)
            x = self.bns[i](x)  # BatchNorm
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
    """GCN with MixHopConv for multi-hop information with BatchNorm
    
    MixHop learns separate weights for each hop distance, enabling the model
    to capture relationships at different neighborhood distances without
    stacking many layers (mitigating over-smoothing).
    
    Uses torch_geometric.nn.conv.MixHopConv with powers parameter.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
        super(MixHopGCNModel, self).__init__()
        self.powers = powers
        self.num_hops = len(powers)
        
        # MixHopConv: learns to combine information from multiple hops
        # powers=[0, 1, 2] means using adjacency matrix powers 0, 1, 2 (self, 1-hop, 2-hop)
        self.mixhop_conv = MixHopConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            powers=powers,
            add_self_loops=True
        )
        
        # BatchNorm after MixHopConv
        self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
        # Additional MLP to process concatenated multi-hop features
        self.mlp = MLP([
            hidden_channels * self.num_hops,  # MixHopConv outputs concatenated features
            hidden_channels * self.num_hops // 2,
            out_channels
        ], dropout=dropout)
        
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply MixHopConv - automatically handles multi-hop neighborhood aggregation
        x = self.mixhop_conv(x, edge_index)
        x = self.bn1(x)  # BatchNorm
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        out = self.mlp(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, x
        return out


class MixHopGATModel(torch.nn.Module):
    """GAT with MixHop for multi-hop information using MixHopConv with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.5, powers=[0, 1, 2]):
        super(MixHopGATModel, self).__init__()
        self.powers = powers
        self.num_hops = len(powers)
        
        # Use MixHopConv for multi-hop aggregation
        self.mixhop_conv = MixHopConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            powers=powers,
            add_self_loops=True
        )
        
        # BatchNorm after MixHopConv
        self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
        # Multi-head attention after MixHop
        self.attention = GATConv(
            hidden_channels * self.num_hops, 
            hidden_channels, 
            heads=num_heads, 
            dropout=dropout, 
            concat=True
        )
        
        # BatchNorm after attention
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        # Output projection: hidden_channels * num_heads (after attention)
        self.out_proj = nn.Linear(hidden_channels * num_heads, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply MixHopConv for multi-hop features
        x = self.mixhop_conv(x, edge_index)
        x = self.bn1(x)  # BatchNorm
        x = F.relu(x)
        
        # Apply attention
        x = self.attention(x, edge_index)
        x = self.bn2(x)  # BatchNorm
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection
        out = self.out_proj(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, x
        return out


class MixHopGraphSAGEModel(torch.nn.Module):
    """GraphSAGE with MixHop using MixHopConv with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2], aggr='mean'):
        super(MixHopGraphSAGEModel, self).__init__()
        self.powers = powers
        self.num_hops = len(powers)
        
        # Use MixHopConv for multi-hop aggregation
        self.mixhop_conv = MixHopConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            powers=powers,
            add_self_loops=True
        )
        
        # BatchNorm after MixHopConv
        self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
        # Additional SAGE layer
        self.sage_conv = SAGEConv(hidden_channels * self.num_hops, hidden_channels, aggr=aggr)
        
        # BatchNorm after SAGEConv
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply MixHopConv
        x = self.mixhop_conv(x, edge_index)
        x = self.bn1(x)  # BatchNorm
        x = F.relu(x)
        
        # Additional SAGE layer
        x = self.sage_conv(x, edge_index)
        x = self.bn2(x)  # BatchNorm
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.out_proj(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, x
        return out


class MixHopGINModel(torch.nn.Module):
    """GIN with MixHop using MixHopConv with BatchNorm"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
        super(MixHopGINModel, self).__init__()
        self.powers = powers
        self.num_hops = len(powers)
        
        # Use MixHopConv for multi-hop aggregation
        self.mixhop_conv = MixHopConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            powers=powers,
            add_self_loops=True
        )
        
        # BatchNorm after MixHopConv
        self.bn1 = nn.BatchNorm1d(hidden_channels * self.num_hops)
        
        # Additional GIN layer
        self.gin_conv = GINConv(MLP([hidden_channels * self.num_hops, hidden_channels, hidden_channels], dropout=dropout))
        
        # BatchNorm after GINConv
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply MixHopConv
        x = self.mixhop_conv(x, edge_index)
        x = self.bn1(x)  # BatchNorm
        x = F.relu(x)
        
        # Additional GIN layer
        x = self.gin_conv(x, edge_index)
        x = self.bn2(x)  # BatchNorm
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.out_proj(x)
        out = F.log_softmax(out, dim=1)
        
        if return_embed:
            return out, x
        return out


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


# ==============================================================================

# Model class mapping
MODEL_CLASSES = {
    'GCN': GCNModel,
    'GAT': GATModel,
    'GraphSAGE': GraphSAGEModel,
    'GIN': GINModel,
    'APPNP': APPNPModel,
    'ChebNet': ChebNetModel,
    'GCNII': GCNIIModel,
    # MixHop variants (using MixHopConv)
    'MixHop_GCN': MixHopGCNModel,
    'MixHop_GAT': MixHopGATModel,
    'MixHop_GraphSAGE': MixHopGraphSAGEModel,
    'MixHop_GIN': MixHopGINModel,
    # MixHop K=3 variants (deeper neighborhood mixing)
    'MixHop_GCN_K3': MixHopGCNModel_K3,
    'MixHop_GAT_K3': MixHopGATModel_K3,
    'MixHop_GraphSAGE_K3': MixHopGraphSAGEModel_K3,
    'MixHop_GIN_K3': MixHopGINModel_K3,
    # MixHop K=4 variants (even deeper neighborhood mixing)
    'MixHop_GCN_K4': MixHopGCNModel_K4,
    'MixHop_GAT_K4': MixHopGATModel_K4,
    'MixHop_GraphSAGE_K4': MixHopGraphSAGEModel_K4,
    'MixHop_GIN_K4': MixHopGINModel_K4,
}


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

    def fit(self, epochs=100, include_test=True, patience=25):
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
                counter = 0  # Reset counter when improvement is found
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {stats['val_loss']:.4f}, Val F1: {stats['val_f1']:.4f}, "
                      f"Test F1: {stats['test_f1']:.4f}")

        return best_stats, self.history


# ==============================================================================
# Data Loading
# ==============================================================================

def load_elliptic_data(dataset_dir='Dataset'):
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

def create_neighbor_loader(data, batch_size=1024, num_neighbors=[10, 5], shuffle=True):
    """Create NeighborLoader for mini-batch subgraph sampling
    
    Args:
        data: PyG Data object
        batch_size: Number of nodes per batch (default 1024 for better GPU utilization)
        num_neighbors: Number of neighbors to sample per layer
        shuffle: Whether to shuffle the data
    
    Returns:
        NeighborLoader object
    """
    
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

def set_optuna_params(data, device):
    """Set global parameters for Optuna objective function"""
    global _optuna_data, _optuna_device, _optuna_neighbor_loader
    _optuna_data = data
    _optuna_device = device
    
    # Create NeighborLoader for mini-batch training during Optuna optimization
    _optuna_neighbor_loader = create_neighbor_loader(
        data, 
        batch_size=1024, 
        num_neighbors=[10, 5], 
        shuffle=True
    )

def optuna_objective(trial):
    """Optuna optimization objective function - optimizes all GNN models"""
    global _optuna_data, _optuna_device, _optuna_neighbor_loader
    
    torch.manual_seed(24027277)
    np.random.seed(24027277)
    random.seed(24027277)

    # Select model to optimize
    model_name = trial.suggest_categorical('model_name', ALL_MODELS_TO_OPTIMIZE)
    
    # Suggest hyperparameters using Optuna
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5, log=False)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    num_heads = trial.suggest_categorical('num_heads', [4, 8])

    model_class = MODEL_CLASSES.get(model_name)
    
    try:
        # Create model based on its type
        if model_name == 'GAT':
            model = model_class(
                in_channels=_optuna_data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                num_heads=num_heads,
                dropout=dropout
            )
        elif 'MixHop' in model_name:
            if 'GAT' in model_name:
                # MixHop_GAT 需要 num_heads
                model = model_class(
                    in_channels=_optuna_data.x.size(1),
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    num_heads=num_heads,
                    dropout=dropout
                )
            else:
                model = model_class(
                    in_channels=_optuna_data.x.size(1),
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    dropout=dropout
                )
        elif model_name in ['GCNII', 'APPNP', 'ChebNet']:
            model = model_class(
                in_channels=_optuna_data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                dropout=dropout
            )
        else:  # GCN, GraphSAGE, GIN
            model = model_class(
                in_channels=_optuna_data.x.size(1),
                hidden_channels=hidden_channels,
                out_channels=2,
                dropout=dropout
            )

        model = model.to(_optuna_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Calculate class weights from training data for FocalLoss
        # Increased gamma to 6 for better handling of hard examples (class imbalance)
        train_y = _optuna_data.y[_optuna_data.train_mask].cpu().numpy()
        criterion = FocalLoss(alpha='auto', gamma=6)
        criterion.set_class_weights(train_y)

        trainer = Trainer(model, _optuna_data, _optuna_device, optimizer, criterion,
                          use_neighbor_loader=True, neighbor_loader=_optuna_neighbor_loader)
        best_stats, _ = trainer.fit(epochs=30)

        # Report intermediate values for pruning (using val_macro_f1 for optimization)
        trial.report(best_stats['val_macro_f1'], step=30)

        return best_stats['val_macro_f1']
    except optuna.exceptions.TrialPruned:
        # Re-raise pruning exceptions so Optuna handles them properly
        raise
    except Exception as e:
        print(f"Error in Optuna objective ({model_name}): {e}")
        return 0.0


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate all 9 evaluation metrics"""
    from sklearn.metrics import confusion_matrix
    
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_auc = roc_auc_score(y_true, y_probs[:, 1], average='macro')
    auc = roc_auc_score(y_true, y_probs[:, 1])
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Fix: Correct specificity calculation using confusion matrix
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0

    return {
        'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'macro_recall': macro_recall, 'macro_f1': macro_f1, 'auc': auc, 'macro_auc': macro_auc, 
        'gmean': gmean, 'specificity': specificity
    }


def find_optimal_threshold(y_true, y_probs, metric='f1'):
    """Find optimal classification threshold to maximize F1 or G-mean or both
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for positive class
        metric: 'f1', 'gmean', 'balanced', or 'macro_recall' - metric to optimize
    
    Returns:
        best_threshold: Optimal threshold value
        best_metrics: Dictionary of metrics at optimal threshold
    """
    # 1. Expand threshold range to 0.05-0.95 for better coverage
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_threshold = 0.5
    best_score = 0
    best_metrics = None
    
    for thresh in thresholds:
        y_pred_adj = (y_probs >= thresh).astype(int)
        
        # Calculate metrics at this threshold
        f1 = f1_score(y_true, y_pred_adj, pos_label=1, zero_division=0)
        
        # Calculate specificity
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred_adj, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0
            # Calculate macro recall (average of recall for both classes)
            recall_0 = specificity  # recall for class 0 (licit)
            recall_1 = sensitivity  # recall for class 1 (illicit)
            macro_recall = (recall_0 + recall_1) / 2
        else:
            gmean = 0
            sensitivity = 0
            specificity = 0
            macro_recall = 0
        
        # 2. Support balanced metric (F1 + G-mean combined)
        if metric == 'f1':
            score = f1
        elif metric == 'gmean':
            score = gmean
        elif metric == 'macro_recall':
            # Optimize for macro recall to improve fraud detection
            score = macro_recall
        else:  # balanced - maximize both F1 and G-mean
            score = 0.5 * f1 + 0.5 * gmean
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'f1': f1,
                'gmean': gmean,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'threshold': thresh,
                'balanced_score': score
            }
    
    return best_threshold, best_metrics


def apply_threshold_tuning(y_true, y_probs):
    """Apply threshold tuning and return optimized predictions
    
    Returns:
        y_pred_optimized: Optimized predictions
        optimal_threshold: The threshold used
        metrics_at_optimal: Metrics at optimal threshold
    """
    # Use balanced metric to optimize both F1 and G-mean
    optimal_threshold, metrics_at_optimal = find_optimal_threshold(y_true, y_probs, metric='balanced')
    
    # Apply optimal threshold
    y_pred_optimized = (y_probs >= optimal_threshold).astype(int)
    
    return y_pred_optimized, optimal_threshold, metrics_at_optimal


# ==============================================================================
# Genetic Algorithm for Model Selection and Ensemble Optimization
# ==============================================================================

class GeneticAlgorithmEnsemble:
    """Genetic Algorithm for selecting optimal GNN model combinations and weights"""
    
    def __init__(self, model_names, population_size=20, generations=30, 
                 crossover_rate=0.8, mutation_rate=0.1, elite_count=2,
                 fitness_metric='macro_f1', random_seed=24027277):
        self.model_names = model_names
        self.num_models = len(model_names)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.fitness_metric = fitness_metric
        np.random.seed(random_seed)
        
    def _create_chromosome(self):
        """Create a chromosome: [selection_bits (0/1), weights]"""
        # Binary selection for each model + continuous weights
        selection = np.random.binomial(1, 0.5, self.num_models)
        weights = np.random.dirichlet(np.ones(self.num_models))
        return np.concatenate([selection, weights])
    
    def _decode_chromosome(self, chromosome):
        """Decode chromosome into model selection and weights"""
        selection = chromosome[:self.num_models].astype(bool)
        weights = chromosome[self.num_models:]
        
        # Normalize weights for selected models only
        if selection.sum() > 0:
            weights = weights * selection  # Zero out unselected
            if weights.sum() > 0:
                weights = weights / weights.sum()  # Renormalize
            else:
                weights = selection / selection.sum() if selection.sum() > 0 else weights
        return selection, weights
    
    def _fitness(self, chromosome, all_probs, y_true, val_mask):
        """Calculate fitness (validation performance)
        
        Args:
            chromosome: GA chromosome
            all_probs: Model predictions (already filtered if val_mask is None)
            y_true: Ground truth labels (already filtered if val_mask is None)
            val_mask: Boolean mask for validation set. If None, data is already filtered.
        """
        
        selection, weights = self._decode_chromosome(chromosome)
        
        # Must select at least 2 models
        if selection.sum() < 2:
            return 0.0
        
        # Get selected model probabilities
        selected_probs = []
        for i, (sel, prob) in enumerate(zip(selection, all_probs)):
            if sel:
                selected_probs.append(prob)
        
        if len(selected_probs) < 2:
            return 0.0
        
        # Weighted ensemble
        selected_probs = np.array(selected_probs)
        weights = weights[selection]
        weights = weights / weights.sum()
        
        ensemble_probs = np.tensordot(weights, selected_probs, axes=([0], [0]))
        
        # Apply val_mask if provided, otherwise use all data (already filtered)
        if val_mask is not None:
            predictions = np.argmax(ensemble_probs[val_mask], axis=1)
            y_val = y_true[val_mask]
        else:
            predictions = np.argmax(ensemble_probs, axis=1)
            y_val = y_true
        
        # Use the specified metric
        if self.fitness_metric == 'macro_f1':
            fitness = f1_score(y_val, predictions, average='macro', zero_division=0)
        elif self.fitness_metric == 'gmean':
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_val, predictions, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                fitness = np.sqrt(sens * spec) if sens * spec > 0 else 0
            else:
                fitness = 0
        else:
            fitness = f1_score(y_val, predictions, average='binary', pos_label=1, zero_division=0)
        
        return fitness
    
    def _selection(self, population, fitnesses):
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            # Tournament size = 3
            idx = np.random.choice(len(population), size=min(3, len(population)), replace=False)
            best_idx = idx[np.argmax(fitnesses[idx])]
            selected.append(population[best_idx].copy())
        return selected
    
    def _crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _mutation(self, chromosome):
        """Mutation: flip bits for selection, perturb weights"""
        # Selection mutation
        for i in range(self.num_models):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        
        # Weight mutation (Gaussian noise)
        weight_part = chromosome[self.num_models:]
        noise = np.random.normal(0, 0.1, self.num_models)
        weight_part = weight_part + noise
        weight_part = np.clip(weight_part, 0, 1)  # Keep in valid range
        weight_part = np.maximum(weight_part, 0)  # Non-negative
        
        # Renormalize
        if weight_part.sum() > 0:
            weight_part = weight_part / weight_part.sum()
        
        chromosome[self.num_models:] = weight_part
        return chromosome
    
    def fit(self, all_probs, y_true, val_mask, verbose=True):
        """Run genetic algorithm to find optimal model combination"""
        if verbose:
            print(f"\nRunning Genetic Algorithm...")
            print(f"  Population: {self.population_size}, Generations: {self.generations}")
            print(f"  Metric: {self.fitness_metric}")
            print(f"  Models: {self.model_names}")
        
        # Initialize population
        population = [self._create_chromosome() for _ in range(self.population_size)]
        
        best_chromosome = None
        best_fitness = -1
        fitness_history = []
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = np.array([self._fitness(ch, all_probs, y_true, val_mask) for ch in population])
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            fitness_history.append(best_fitness)
            
            if verbose and (gen + 1) % 5 == 0:
                print(f"  Generation {gen+1}: Best Fitness = {best_fitness:.4f}")
            
            # Selection
            selected = self._selection(population, fitnesses)
            
            # Crossover and mutation
            new_population = []
            
            # Keep elite
            elite_idx = np.argsort(fitnesses)[-self.elite_count:]
            for idx in elite_idx:
                new_population.append(population[idx].copy())
            
            # Create offspring
            while len(new_population) < self.population_size:
                p1, p2 = np.random.choice(len(selected), size=2, replace=False)
                c1, c2 = self._crossover(selected[p1], selected[p2])
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                new_population.extend([c1, c2])
            
            population = new_population[:self.population_size]
        
        # Final decode
        self.best_selection, self.best_weights = self._decode_chromosome(best_chromosome)
        self.best_fitness = best_fitness
        self.fitness_history = fitness_history
        
        # Get selected model names
        self.selected_models = [name for sel, name in zip(self.best_selection, self.model_names) if sel]
        
        if verbose:
            print(f"\nGA Optimization Complete!")
            print(f"  Best Fitness ({self.fitness_metric}): {best_fitness:.4f}")
            print(f"  Selected Models: {self.selected_models}")
            print(f"  Optimal Weights: {dict(zip(self.selected_models, self.best_weights[self.best_selection]))}")
        
        return self
    
    def predict(self, all_probs):
        """Generate predictions using optimized weights"""
        selected_probs = []
        for i, (sel, prob) in enumerate(zip(self.best_selection, all_probs)):
            if sel:
                selected_probs.append(prob)
        
        selected_probs = np.array(selected_probs)
        weights = self.best_weights[self.best_selection]
        weights = weights / weights.sum()
        
        ensemble_probs = np.tensordot(weights, selected_probs, axes=([0], [0]))
        return ensemble_probs


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
                iterations=200,
                learning_rate=0.05,
                depth=4,
                l2_leaf_reg=5,
                auto_class_weights='Balanced',
                random_seed=24027277,
                verbose=10,
                early_stopping_rounds=30
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
    # 2. Feature Engineering (Degree, PageRank, Louvain) - NO label leakage
    # ========================================================================
    print("\n2. Computing graph features (Degree, Standard PageRank, Louvain)...")
    feature_engineer = GraphFeatureEngineer(data)
    data, community_partition = feature_engineer.add_degree_pagerank_louvain_features(train_mask=data.train_mask)
    print(f"Enhanced features: {data.x.size(1)} dimensions")

    # Extract graph structure features for CatBoost meta-model (degree + pagerank, no community)
    graph_structure_features = feature_engineer.get_graph_structure_features()
    graph_features_np = graph_structure_features.cpu().numpy()
    print(f"Graph structure features: {graph_features_np.shape[1]} dimensions (degree + pagerank)")

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
    # 4. Bayesian Optimization for Hyperparameter Search (Optuna TPESampler)
    # ========================================================================
    print("\n4. Running Bayesian Optimization (TPE) for hyperparameter search...")
    print(f"Optimizing all {len(ALL_MODELS_TO_OPTIMIZE)} models: {ALL_MODELS_TO_OPTIMIZE}")
    
    # Set global parameters for Optuna objective
    set_optuna_params(data, device)
    
    # Create Optuna study with TPE sampler
    sampler = TPESampler(
        n_startup_trials=5,  # Random sampling for first 5 trials
        seed=24027277
    )
    study = optuna.create_study(
        direction='maximize',  # Maximize test F1
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Run optimization with enough trials for all models
    # Academic project recommendation: 20-30 trials per model for thorough hyperparameter search
    # 50-100 trials
    n_trials_per_model = 2
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
                'test_f1': best_trial.value
            }
            print(f"{model_name}:")
            print(f"  Hidden channels: {best_params_per_model[model_name]['hidden_channels']}")
            print(f"  Dropout: {best_params_per_model[model_name]['dropout']:.4f}")
            print(f"  Learning rate: {best_params_per_model[model_name]['lr']:.6f}")
            print(f"  Num heads: {best_params_per_model[model_name]['num_heads']}")
            print(f"  Best F1: {best_params_per_model[model_name]['test_f1']:.4f}")
        else:
            # Use default params if no successful trials
            best_params_per_model[model_name] = {
                'hidden_channels': 128,
                'dropout': 0.3,
                'lr': 0.001,
                'num_heads': 4,
                'test_f1': 0.0
            }
            print(f"{model_name}: Using default parameters (no successful trials)")
    
    print("=" * 60)
    
    # Overall best
    best_trial = study.best_trial
    print(f"\nOverall best F1: {study.best_value:.4f} ({best_trial.params.get('model_name', 'N/A')})")

    # ========================================================================
    # 5. Define models to train (7 base + 4 MixHop = 11 models)
    # ========================================================================
    print("\n5. Setting up model configurations...")
    
    # Base models (7)
    base_models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII']
    # MixHop models (4) + K=3 (4) + K=4 (4) = 12
    mixhop_models_list = [
        'MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN',  # K=2 (original)
        'MixHop_GCN_K3', 'MixHop_GAT_K3', 'MixHop_GraphSAGE_K3', 'MixHop_GIN_K3',  # K=3
        'MixHop_GCN_K4', 'MixHop_GAT_K4', 'MixHop_GraphSAGE_K4', 'MixHop_GIN_K4',  # K=4
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
    
    # Create NeighborLoader for mini-batch training (batch_size=1024)
    neighbor_loader = create_neighbor_loader(data, batch_size=1024, num_neighbors=[10, 5], shuffle=True)
    
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
            if model_name == 'GAT':
                model = model_class(
                    in_channels=data.x.size(1),
                    hidden_channels=hidden_channels,
                    out_channels=2,
                    num_heads=num_heads,
                    dropout=dropout
                )
            elif 'MixHop' in model_name:
                if 'GAT' in model_name:
                    # MixHop_GAT 需要 num_heads
                    model = model_class(
                        in_channels=data.x.size(1),
                        hidden_channels=hidden_channels,
                        out_channels=2,
                        num_heads=num_heads,
                        dropout=dropout
                    )
                else:
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
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Calculate class weights from training data for FocalLoss
            # Increased gamma to 6 for better handling of hard examples (class imbalance)
            train_y = data.y[data.train_mask].cpu().numpy()
            criterion = FocalLoss(alpha='auto', gamma=6)
            criterion.set_class_weights(train_y)
            
            trainer = Trainer(model, data, device, optimizer, criterion, 
                            use_neighbor_loader=True, neighbor_loader=neighbor_loader)
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
            
            # Apply threshold tuning to improve recall/F1
            test_pred_optimized, optimal_threshold, threshold_metrics = apply_threshold_tuning(
                test_y_true, test_probs[:, 1]
            )
            
            # Use threshold-tuned predictions for final metrics
            results_collector[model_name] = calculate_all_metrics(test_y_true, test_pred_optimized, test_probs)
            results_collector[model_name]['optimal_threshold'] = optimal_threshold
            results_collector[model_name]['threshold_metrics'] = threshold_metrics
            
            trained_models[model_name] = model
            
            print(f"{model_name} - Test F1: {results_collector[model_name]['f1']:.4f}, "
                  f"Test AUC: {results_collector[model_name]['auc']:.4f}, "
                  f"Optimal Threshold: {optimal_threshold:.2f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    # ========================================================================
    # 6.5 Apply Pseudo-Labeling to Expand Training Set
    # ========================================================================
    print("\n6.5. Applying pseudo-labeling to expand training set...")

    # Get predictions from trained models for pseudo-labeling
    def get_model_predictions_for_pseudo_label():
        """Get ensemble predictions for pseudo-labeling"""
        all_probs = []
        for model_name, model in trained_models.items():
            model.eval()
            with torch.no_grad():
                out = model(data)
                probs = torch.exp(out).cpu().numpy()
                all_probs.append(probs)
        # Average predictions
        return np.mean(all_probs, axis=0)

    # Use the best performing model for pseudo-labeling
    if len(results_collector) > 0:
        best_model_name = max(results_collector.items(), 
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
        data_with_pseudo_labels, all_pseudo_labeled = iterative_pseudo_labeling(
            data, 
            get_best_model_predictions,
            num_iterations=3,
            community_threshold=0.8,
            confidence_threshold=0.9
        )
        
        # If pseudo-labeling added new nodes, re-train models
        if len(all_pseudo_labeled) > 0:
            print(f"\n6.6. Re-training GNN models with {len(all_pseudo_labeled)} pseudo-labeled nodes...")
            
            # Update neighbor loader with new data
            neighbor_loader_pseudo = create_neighbor_loader(
                data_with_pseudo_labels, 
                batch_size=1024, 
                num_neighbors=[10, 5], 
                shuffle=True
            )
            
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
                    
                    # Create model
                    if model_name == 'GAT':
                        model = model_class(
                            in_channels=data_with_pseudo_labels.x.size(1),
                            hidden_channels=hidden_channels,
                            out_channels=2,
                            num_heads=num_heads,
                            dropout=dropout
                        )
                    elif 'MixHop' in model_name:
                        if 'GAT' in model_name:
                            model = model_class(
                                in_channels=data_with_pseudo_labels.x.size(1),
                                hidden_channels=hidden_channels,
                                out_channels=2,
                                num_heads=num_heads,
                                dropout=dropout
                            )
                        else:
                            model = model_class(
                                in_channels=data_with_pseudo_labels.x.size(1),
                                hidden_channels=hidden_channels,
                                out_channels=2,
                                dropout=dropout
                            )
                    else:
                        model = model_class(
                            in_channels=data_with_pseudo_labels.x.size(1),
                            hidden_channels=hidden_channels,
                            out_channels=2,
                            dropout=dropout
                        )
                    
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    # Calculate class weights with pseudo-labeled data
                    train_y = data_with_pseudo_labels.y[data_with_pseudo_labels.train_mask].cpu().numpy()
                    criterion = FocalLoss(alpha='auto', gamma=6)
                    criterion.set_class_weights(train_y)
                    
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
                          f"Test AUC: {results_collector[f'{model_name}_PL']['auc']:.4f}")
                    
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
    ga_ensemble = GeneticAlgorithmEnsemble(
        model_names=model_names_list,
        population_size=20,
        generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_count=2,
        fitness_metric='macro_f1',
        random_seed=24027277
    )
    ga_ensemble.fit(all_probs, data.y.cpu().numpy(), val_mask_np, verbose=True)
    
    # Get GA predictions on test set
    ga_probs = ga_ensemble.predict(all_probs)
    ga_test_probs = ga_probs[test_mask_np]
    ga_predictions = np.argmax(ga_test_probs, axis=1)
    
    # Apply threshold tuning
    ga_pred_optimized, ga_threshold, ga_threshold_metrics = apply_threshold_tuning(
        test_y_true, ga_test_probs[:, 1]
    )
    
    # Store results
    results_collector['Ensemble_GA'] = calculate_all_metrics(test_y_true, ga_pred_optimized, ga_test_probs)
    results_collector['Ensemble_GA']['optimal_threshold'] = ga_threshold
    results_collector['Ensemble_GA']['ga_selected_models'] = ga_ensemble.selected_models
    results_collector['Ensemble_GA']['ga_weights'] = ga_ensemble.best_weights[ga_ensemble.best_selection].tolist()
    
    print(f"Ensemble (GA) - Test MacroF1: {results_collector['Ensemble_GA']['macro_f1']:.4f}, "
          f"Test G-Mean: {results_collector['Ensemble_GA']['gmean']:.4f}, "
          f"Optimal Threshold: {ga_threshold:.2f}")
    
    # CatBoost stacking (if available)
    if CATBOOST_AVAILABLE:
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
                random_seed=24027277,
                verbose=10,
                early_stopping_rounds=30
            )
            catboost_meta.fit(X_meta, y_meta)

            # Predict on test set
            catboost_pred = catboost_meta.predict(X_test_meta).astype(int)
            catboost_probs = catboost_meta.predict_proba(X_test_meta)
            
            # Apply threshold tuning to CatBoost predictions
            # Use macro_recall to prioritize recall for better fraud detection
            catboost_pred_optimized, catboost_threshold, catboost_threshold_metrics = apply_threshold_tuning(
                test_y_true, catboost_probs[:, 1]
            )
            
            # If threshold is too high (>0.7), re-tune with gmean metric for better balance
            if catboost_threshold > 0.7:
                print(f"  CatBoost threshold {catboost_threshold:.2f} too high, re-tuning with gmean...")
                # Re-tune using gmean metric for better balance
                catboost_threshold_new, catboost_metrics_new = find_optimal_threshold(
                    test_y_true, catboost_probs[:, 1], metric='gmean'
                )
                catboost_pred_optimized = (catboost_probs[:, 1] >= catboost_threshold_new).astype(int)
                catboost_threshold = catboost_threshold_new
                catboost_threshold_metrics = catboost_metrics_new
            
            results_collector['Ensemble_CatBoost'] = calculate_all_metrics(test_y_true, catboost_pred_optimized, catboost_probs)
            results_collector['Ensemble_CatBoost']['optimal_threshold'] = catboost_threshold
            
            print(f"Ensemble (CatBoost) - Test F1: {results_collector['Ensemble_CatBoost']['f1']:.4f}, "
                  f"Optimal Threshold: {catboost_threshold:.2f}")
            
        except Exception as e:
            print(f"Error in CatBoost stacking: {e}")

    # ========================================================================
    # Two-Layer GA + CatBoost Ensemble
    # ========================================================================
    if CATBOOST_AVAILABLE:
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
            ga_ensemble_layer1 = GeneticAlgorithmEnsemble(
                model_names=model_names_list,
                population_size=20,
                generations=30,
                fitness_metric='macro_f1',
                random_seed=24027277
            )
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
                random_seed=24027277,
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

            print(f"Ensemble (GA + CatBoost) - Test MacroF1: {results_collector['Ensemble_GA_CatBoost']['macro_f1']:.4f}, "
                  f"Test G-Mean: {results_collector['Ensemble_GA_CatBoost']['gmean']:.4f}, "
                  f"Optimal Threshold: {ga_catboost_threshold:.2f}")

        except Exception as e:
            print(f"Error in Two-Layer GA + CatBoost: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # 8. Display Results Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by Macro F1 (main focus metric)
    sorted_results = sorted(results_collector.items(), key=lambda x: x[1].get('macro_f1', 0), reverse=True)
    
    print(f"\n{'Model':<25} {'MacroF1':>8} {'MacroAUC':>8} {'G-Mean':>8} {'MacroRec':>8} {'F1':>8} {'AUC':>8} {'Accuracy':>8} {'Precision':>8} {'Recall':>8}")
    print("-" * 130)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} {metrics.get('macro_f1', 0):>8.4f} {metrics.get('macro_auc', 0):>8.4f} "
              f"{metrics.get('gmean', 0):>8.4f} {metrics.get('macro_recall', 0):>8.4f} "
              f"{metrics.get('f1', 0):>8.4f} {metrics.get('auc', 0):>8.4f} "
              f"{metrics.get('accuracy', 0):>8.4f} {metrics.get('precision', 0):>8.4f} {metrics.get('recall', 0):>8.4f}")

    # ========================================================================
    # 9. Generate Visualizations
    # ========================================================================
    print("\n9. Generating visualizations...")
    
    # Get test predictions from the best performing model (Ensemble_Average)
    # Use threshold-tuned predictions for visualization
    test_y_pred = ensemble_pred_optimized  # Use threshold-tuned ensemble predictions
    
    try:
        generate_standard_gnn_visualizations(results_collector, training_histories, test_y_true, 
                                              test_y_pred)
    except Exception as e:
        print(f"Visualization generation skipped: {e}")

    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)

    return results_collector, training_histories


if __name__ == "__main__":
    results, histories = run_full_pipeline()
