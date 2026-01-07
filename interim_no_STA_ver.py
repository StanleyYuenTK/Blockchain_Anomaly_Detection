"""
Student ID: 24027277d
Name: Yuen Tsz Ki
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP

from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial
from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations


def create_model_configs(data, best_hidden_channels, best_num_heads, dropout_value):
    """Create model parameter configurations for all GNN models"""
    base_config = {
        'in_channels': data.x.size(1),
        'hidden_channels': best_hidden_channels,
        'out_channels': 2,
        # 'num_layers': 2,  # Fixed to 2 layers
        'dropout': dropout_value
    }

    configs = {
        'GCN': base_config.copy(),
        'GIN': base_config.copy(),
        'GraphSAGE': base_config.copy(),
        'GAT': {
            **base_config,
            'num_heads': best_num_heads
        }
    }

    return configs


# -----------------------
# Loss Functions
# -----------------------

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------
# Model definitions
# -----------------------

class BaseGNN(nn.Module):
    """Provide unified forward(self, data, return_embed=False) interface"""
    def forward(self, data, return_embed=False):
        raise NotImplementedError


class GCNModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCNModel, self).__init__()
        self.hidden_channels = hidden_channels
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
            # Return the last hidden representation (if no hidden layers, return input)
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GATModel(BaseGNN):
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


class GraphSAGEModel(BaseGNN):
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


class GINModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GINModel, self).__init__()
        self.convs = nn.ModuleList()
        # mlp_channels = [in_channels, hidden_channels, hidden_channels]
        self.convs.append(GINConv(MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)))
        # mlp_channels = [hidden_channels, hidden_channels, out_channels]
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


# -----------------------
# GAN Data Augmentation
# -----------------------

class GAN_Generator(nn.Module):
    """Simple GAN Generator for data augmentation"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAN_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, z):
        return self.model(z)


class GAN_Discriminator(nn.Module):
    """Simple GAN Discriminator for data augmentation"""
    def __init__(self, input_dim, hidden_dim):
        super(GAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Model class mapping for cleaner code
MODEL_CLASSES = {
    'GCN': GCNModel,
    'GAT': GATModel,
    'GIN': GINModel,
    'GraphSAGE': GraphSAGEModel
}


def augment_illegal_samples_with_gan(data, device, latent_dim=100, hidden_dim=128, num_epochs=100, augment_ratio=0.5):
    """Use GAN to generate additional illegal (class 1) samples"""

    # Extract illegal samples (class 1)
    illegal_mask = (data.y == 1)
    illegal_features = data.x[illegal_mask]
    num_illegal = illegal_features.size(0)
    num_to_generate = int(num_illegal * augment_ratio)

    if num_illegal == 0:
        print("No illegal samples found, skipping GAN augmentation")
        return data

    print(f"Found {num_illegal} illegal samples, will generate {num_to_generate} additional samples")

    # Initialize GAN models
    feature_dim = data.x.size(1)
    generator = GAN_Generator(latent_dim, hidden_dim, feature_dim).to(device)
    discriminator = GAN_Discriminator(feature_dim, hidden_dim).to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training data for discriminator
    real_labels = torch.ones(num_illegal, 1).to(device)
    fake_labels = torch.zeros(num_illegal, 1).to(device)

    # Training loop
    for epoch in range(num_epochs):
        # Train Discriminator
        d_optimizer.zero_grad()

        # Real samples
        real_output = discriminator(illegal_features)
        d_loss_real = criterion(real_output, real_labels)

        # Fake samples
        z = torch.randn(num_illegal, latent_dim).to(device)
        fake_features = generator(z)
        fake_output = discriminator(fake_features.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_features)
        g_loss = criterion(fake_output, real_labels)  # Generator wants discriminator to think fake is real
        g_loss.backward()
        g_optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"GAN Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Generate new samples
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_to_generate, latent_dim).to(device)
        generated_features = generator(z)

        # Add small noise to make samples more diverse
        noise = torch.randn_like(generated_features) * 0.1
        generated_features = generated_features + noise

    # Create new data with augmented samples (keep all tensors on the same device)
    new_x = torch.cat([data.x, generated_features], dim=0)
    new_y = torch.cat([data.y, torch.ones(num_to_generate, dtype=torch.long).to(device)], dim=0)

    # Create new masks (keep original train/val/test split, add new samples to training)
    new_train_mask = torch.cat([data.train_mask, torch.ones(num_to_generate, dtype=torch.bool).to(device)], dim=0)
    new_val_mask = torch.cat([data.val_mask, torch.zeros(num_to_generate, dtype=torch.bool).to(device)], dim=0)
    new_test_mask = torch.cat([data.test_mask, torch.zeros(num_to_generate, dtype=torch.bool).to(device)], dim=0)

    # Create augmented data object
    augmented_data = Data(
        x=new_x,
        y=new_y,
        edge_index=data.edge_index,  # Keep original edges
        train_mask=new_train_mask,
        val_mask=new_val_mask,
        test_mask=new_test_mask
    )

    # Move augmented data back to the specified device
    augmented_data = augmented_data.to(device)

    print(f"GAN augmentation completed: added {num_to_generate} synthetic illegal samples")
    return augmented_data

# -----------------------
# Isolation Forest Baseline
# -----------------------

class IsolationForestBaseline:
    """Baseline model using Isolation Forest for anomaly detection"""

    def __init__(self, n_estimators=100, max_samples='auto', random_state=42, contamination='auto'):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.model = None
        self.scaler = None

    def fit(self, data):
        """Train Isolation Forest on node features only (ignore graph structure)"""
        print("Training Isolation Forest baseline model...")

        # Extract training data (only node features, ignore edge_index)
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()

        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Calculate contamination if set to 'auto'
        if self.contamination == 'auto':
            # Use the proportion of illegal transactions (class 1) as contamination
            contamination_ratio = np.mean(y_train == 1)
            # Cap at 0.5 as required by sklearn's IsolationForest (must be <= 0.5)
            self.contamination = min(max(contamination_ratio, 0.01), 0.5)  # Min 1%, Max 50%
            

        # Initialize and train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )

        self.model.fit(X_train_scaled)
        print("Isolation Forest training completed")

        return self

    def predict(self, data):
        """Predict on test data"""
        X_test = data.x[data.test_mask].cpu().numpy()
        X_test_scaled = self.scaler.transform(X_test)

        # Get anomaly scores (-1 for anomalies, 1 for normal)
        scores = self.model.decision_function(X_test_scaled)
        predictions = self.model.predict(X_test_scaled)

        # Convert to binary classification (1 for illegal/anomaly, 0 for legal/normal)
        # Isolation Forest: -1 = anomaly (illegal), 1 = normal (legal)
        binary_predictions = (predictions == -1).astype(int)

        return binary_predictions, scores

    def evaluate(self, data):
        """Evaluate baseline model using the same metrics as GNN models"""
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred, anomaly_scores = self.predict(data)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # Macro metrics
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # AUC (using anomaly scores)
        # Convert anomaly scores to probability-like scores for AUC
        # Isolation Forest scores: negative = more anomalous
        # Convert to positive scores where higher = more likely illegal
        prob_scores = -anomaly_scores  # Flip sign so higher = more anomalous
        auc = roc_auc_score(y_true, prob_scores)
        macro_auc = auc  # For binary classification, macro AUC equals regular AUC

        # G-Mean
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
            'macro_auc': macro_auc,
            'gmean': gmean,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }

        return results

# -----------------------
# Data Loading
# -----------------------

def load_elliptic_data(dataset_dir='../Dataset'):
    """
    Load Elliptic Bitcoin transaction dataset.

    Dataset Description:
    - 203,769 nodes (transactions), 234,355 edges
    - Each node has 166 features (after excluding txId, timestep, and class)
    - Labels: '2'/'licit' -> 0 (licit), '1'/'illicit' -> 1 (illicit), 'unknown' -> -1
    - Time steps: 1-49 (about 2 weeks each)
    - Known labels: 4,545 illicit (2%), 42,019 licit (21%), rest unknown

    Returns:
        PyTorch Geometric Data object with x, y, edge_index, and masks
    """
    print("Loading Elliptic dataset...")

    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')

    # Load data
    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)

    # Rename columns for clarity
    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
    features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(2, features_df.shape[1])]

    # Merge features with classes
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')

    # Extract features (exclude txId, timestep, and class columns)
    # Should result in 166 features as per dataset description
    feature_columns = nodes_df.columns[2:-1]  # Skip txId (0), timestep (1), and class (last)
    features = nodes_df[feature_columns].values
    x = torch.tensor(features, dtype=torch.float)

    print(f"Loaded {x.size(0)} nodes with {x.size(1)} features (expected: 166)")

    # Convert labels: '2'/'licit' -> 0 (licit), '1'/'illicit' -> 1 (illicit), else -> -1 (Unknown)
    labels = nodes_df['class'].apply(lambda c: 0 if c == '2' else (1 if c == '1' else -1))
    y = torch.tensor(labels.values, dtype=torch.long)

    # Print label statistics
    total_nodes = len(y)
    licit_count = (y == 0).sum().item()
    illicit_count = (y == 1).sum().item()
    unknown_count = (y == -1).sum().item()

    print(f"Label distribution:")
    print(f"class1 (Illicit): {illicit_count} nodes ({illicit_count/total_nodes*100:.1f}%)")
    print(f"class2 (licit): {licit_count} nodes ({licit_count/total_nodes*100:.1f}%)")
    print(f"class-1 (Unknown): {unknown_count} nodes ({unknown_count/total_nodes*100:.1f}%)")
    # Build transaction ID mapping
    tx_ids = nodes_df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    # Build edge index
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

    # Extract timesteps
    timesteps = torch.tensor(nodes_df['timestep'].values, dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps

    # Create masks for train/val/test splits
    # Time-based split: timesteps 1-34 (train), 35-41 (val), 42-49 (test)
    known_mask = y != -1  # Only use known labels
    data.train_mask = (timesteps < 35) & known_mask
    data.val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    data.test_mask = (timesteps >= 42) & known_mask

    # Print split statistics
    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()

    print(f"Data splits (known labels only):")
    print(f"  Train: {train_count} nodes (timesteps 1-34)")
    print(f"  Validation: {val_count} nodes (timesteps 35-41)")
    print(f"  Test: {test_count} nodes (timesteps 42-49)")

    # Print class distribution in test set
    test_labels = y[data.test_mask]
    test_licit = (test_labels == 0).sum().item()
    test_illicit = (test_labels == 1).sum().item()
    print(f"Test set class distribution:")
    print(f"  licit: {test_licit} ({test_licit/test_count*100:.1f}%)")
    print(f"  illicit: {test_illicit} ({test_illicit/test_count*100:.1f}%)")

    return data



# -----------------------
# Trainer Class
# -----------------------

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

        # Validation set macro metrics
        val_macro_recall = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_macro_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)

        # Validation set AUC
        val_probs = torch.exp(out).cpu().numpy()
        val_mask_np = self.data.val_mask.cpu().numpy()
        val_auc = roc_auc_score(val_y_true, val_probs[val_mask_np][:, 1])
        val_macro_auc = val_auc  # For binary classification, macro AUC equals regular AUC

        # Validation set G-Mean
        val_sensitivity = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
        val_specificity = recall_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
        val_gmean = np.sqrt(val_sensitivity * val_specificity) if val_sensitivity * val_specificity > 0 else 0

        result = {
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_macro_recall': val_macro_recall,
            'val_macro_f1': val_macro_f1,
            'val_macro_auc': val_macro_auc,
            'val_gmean': val_gmean
        }

        # Test set evaluation (only if requested and test samples exist)
        if include_test and self.data.test_mask.sum() > 0:
            test_y_true = self.data.y[self.data.test_mask].cpu().numpy()
            test_y_pred = pred[self.data.test_mask].cpu().numpy()
            test_f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
            test_acc = accuracy_score(test_y_true, test_y_pred)

            # Calculate probabilities for AUC
            probs = torch.exp(out).cpu().numpy()
            test_mask_np = self.data.test_mask.cpu().numpy()

            # Illicit class metrics (Class 1)
            test_precision_illicit = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_recall_illicit = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)

            # Macro metrics
            test_macro_recall = recall_score(test_y_true, test_y_pred, average='macro', zero_division=0)
            test_macro_f1 = f1_score(test_y_true, test_y_pred, average='macro', zero_division=0)

            # AUC
            test_auc = roc_auc_score(test_y_true, probs[test_mask_np][:, 1])
            test_macro_auc = test_auc  # For binary classification, macro AUC equals regular AUC

            # G-Mean
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
                'test_macro_auc': test_macro_auc,
                'test_gmean': test_gmean
            })
        else:
            # Add dummy test metrics if no test data available
            result.update({
                'test_f1': 0.0,
                'test_acc': 0.0,
                'test_precision_illicit': 0.0,
                'test_recall_illicit': 0.0,
                'test_macro_recall': 0.0,
                'test_macro_f1': 0.0,
                'test_auc': 0.0,
                'test_macro_auc': 0.0,
                'test_gmean': 0.0
            })

        return result

    def fit(self, epochs=100, include_test=True):
        best_val_loss = float('inf')
        best_stats = None

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            stats = self.evaluate(include_test=include_test)

            self.history.add_epoch(
                epoch=epoch + 1,  # Epoch starts from 1
                train_loss=train_loss,
                val_loss=stats['val_loss'],
                val_f1=stats['val_f1'],
                test_f1=stats['test_f1'],
                val_acc=stats['val_acc'],
                test_acc=stats['test_acc'],
                val_macro_recall=stats['val_macro_recall'],
                test_macro_recall=stats['test_macro_recall'],
                val_gmean=stats['val_gmean'],
                test_gmean=stats['test_gmean'],
                val_macro_f1=stats['val_macro_f1'],
                test_macro_f1=stats['test_macro_f1'],
                val_macro_auc=stats['val_macro_auc'],  # For binary classification, macro AUC equals regular AUC
                test_macro_auc=stats['test_macro_auc']
            )

            if stats['val_loss'] < best_val_loss:
                best_val_loss = stats['val_loss']
                best_stats = stats

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {stats['val_loss']:.4f}, Val F1: {stats['val_f1']:.4f}, "
                      f"Test F1: {stats['test_f1']:.4f}")


        return best_stats, self.history


class FinalEnsemble:
    """Final ensemble combining multiple single models"""
    def __init__(self, models):
        self.models = models  # dict of single trained models

    def predict(self, data, device, test_only=False):
        all_predictions = []
        all_probs = []

        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                output = model(data)
                probs = torch.exp(output).cpu().numpy()
                pred = np.argmax(probs, axis=1)
                all_predictions.append(pred)
                all_probs.append(probs)

        # Final soft voting
        final_avg_probs = np.mean(all_probs, axis=0)
        final_pred = np.argmax(final_avg_probs, axis=1)

        if test_only:
            # Return only test predictions
            test_mask_np = data.test_mask.cpu().numpy()
            return final_pred[test_mask_np]
        else:
            return final_pred

    def predict_proba(self, data, device, test_only=False):
        """Return averaged probabilities from all models"""
        all_probs = []

        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                output = model(data)
                probs = torch.exp(output).cpu().numpy()
                all_probs.append(probs)

        # Final soft voting across all model types
        final_avg_probs = np.mean(all_probs, axis=0)

        if test_only:
            # Return only test probabilities
            test_mask_np = data.test_mask.cpu().numpy()
            return final_avg_probs[test_mask_np]
        else:
            return final_avg_probs





# -----------------------
# Hyperopt Objective Function
# -----------------------

def hyperopt_objective(data, device, params):
    """Hyperopt optimization objective function for GNN hyperparameter tuning.

    Args:
        data: PyTorch Geometric Data object containing the graph dataset
        device: PyTorch device (CPU or GPU) for model training
        params: Dictionary containing hyperparameters from hyperopt search space

    Returns:
        dict: Dictionary with loss value and additional metrics for hyperopt
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Hyperopt returns actual values, not indices
    model_name = params['model']
    hidden_channels = params['hidden_channels']
    dropout = params['dropout']
    num_heads = params['num_heads']

    # Create model using the mapping (all models use 2 layers)
    model_class = MODEL_CLASSES.get(model_name, GCNModel)
    if model_name == 'GAT':
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

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = FocalLoss(alpha=0.25, gamma=2)  # Use Focal Loss for class imbalance (alpha < 1 for minority class)

    trainer = Trainer(model, data, device, optimizer, criterion)

    # Train
    best_stats, _ = trainer.fit(epochs=50)  # Fewer training epochs to speed up optimization, ignore history

    # Return negative F1 score (Hyperopt minimizes objective)
    return {
        'loss': -best_stats['test_f1'],
        'status': STATUS_OK,
        'test_f1': best_stats['test_f1'],
        'test_macro_f1': best_stats['test_macro_f1'],
        'test_macro_auc': best_stats['test_macro_auc'],
        'test_gmean': best_stats['test_gmean']
    }


# -----------------------
# Evaluation Functions
# -----------------------

def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate all 9 evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_probs: Predicted probabilities for positive class (numpy array)

    Returns:
        dict: Dictionary containing all 9 metrics:
            - f1: F1 score for positive class
            - accuracy: Overall accuracy
            - precision: Precision for positive class
            - recall: Recall for positive class
            - macro_recall: Macro-averaged recall
            - macro_f1: Macro-averaged F1 score
            - auc: Area under ROC curve
            - macro_auc: Macro-averaged AUC (same as auc for binary)
            - gmean: Geometric mean of sensitivity and specificity
    """
    # Basic metrics
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Macro metrics
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC
    auc = roc_auc_score(y_true, y_probs[:, 1])
    macro_auc = auc  # For binary classification, macro AUC equals regular AUC

    # G-Mean
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0

    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'auc': auc,
        'macro_auc': macro_auc,
        'gmean': gmean
    }


# -----------------------
# Main Execution Function
# -----------------------

def run_full_pipeline():
    """Execute complete GNN anomaly detection pipeline"""
    print("=" * 60)
    print("Blockchain Anomaly Detection GNN Framework - Complete Pipeline")
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
    print(f"Data loading completed: {data.x.size(0)} nodes, {data.x.size(1)} features, {data.edge_index.size(1)} edges")

    # ========================================================================
    # 2. Train Isolation Forest baseline
    # ========================================================================
    print("\n2. Training Isolation Forest baseline model...")
    baseline_model = IsolationForestBaseline(
        n_estimators=100,
        contamination='auto',  # Auto-calculate from training data
        random_state=42
    )
    baseline_model.fit(data)
    baseline_results = baseline_model.evaluate(data)
    print("Baseline model evaluation completed")

    # ========================================================================
    # 3. Apply GAN-based data augmentation
    # ========================================================================
    print("\n3. Applying GAN-based data augmentation...")
    data = augment_illegal_samples_with_gan(data, device, augment_ratio=0.3, num_epochs=50)  # Generate 30% more illegal samples
    print(f"Data augmentation completed: {data.x.size(0)} nodes total")

    # ========================================================================
    # 4. Hyperparameter optimization
    # ========================================================================
    print("\n4. Starting Hyperopt hyperparameter optimization...")

    # Define search space with reasonable minimum values
    space = {
        'model': hp.choice('model', ['GCN', 'GAT', 'GIN', 'GraphSAGE']),
        'hidden_channels': hp.choice('hidden_channels', [64, 128, 256]),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'num_heads': hp.choice('num_heads', [4, 8])
    }

    # Define choice lists for index conversion
    MODEL_CHOICES = ['GCN', 'GAT', 'GIN', 'GraphSAGE']
    HIDDEN_CHANNEL_CHOICES = [64, 128, 256]
    NUM_HEAD_CHOICES = [4, 8]

    # Run optimization
    trials = Trials()
    # Use partial to bind data and device to the objective function
    objective_with_data = partial(hyperopt_objective, data, device)
    best = fmin(
        objective_with_data,
        space=space,
        algo=tpe.suggest,
        max_evals=4,  # set 1 for my computer performance
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Convert indices back to actual values (hp.choice returns indices, not values)
    best_model = MODEL_CHOICES[best['model']]
    best_hidden_channels = HIDDEN_CHANNEL_CHOICES[best['hidden_channels']]
    best_num_heads = NUM_HEAD_CHOICES[best['num_heads']]

    # Validate parameters to ensure they're reasonable
    best_hidden_channels = max(best_hidden_channels, 64)  # Ensure at least 64
    best_num_heads = max(min(best_num_heads, 8), 4)       # Ensure between 4 and 8

    print("Best hyperparameters:")
    print(f"Model: {best_model}")
    print(f"Hidden channels: {best_hidden_channels}")
    print(f"Dropout: {best['dropout']:.3f}")
    print(f"Layers: 2 (fixed)")
    print(f"Learning rate: {best['lr']:.6f}")
    print(f"Num heads: {best_num_heads}")

    # Validate final parameters
    print(f"Final model config: {best_hidden_channels} hidden, 2 layers, {best_num_heads} heads")

    # ========================================================================
    # 5. Train individual GNN models (single models)
    # ========================================================================
    print("\n5. Training individual GNN models...")

    # Initialize results collector and training histories
    results_collector = {
        'IForest': baseline_results,
        'GCN_Single': {},
        'GAT_Single': {},
        'GIN_Single': {},
        'GraphSAGE_Single': {},
        'Final_Ensemble': {}
    }

    training_histories = {}  # Store training histories for visualization

    # Define test labels for evaluation
    test_y_true = data.y[data.test_mask].cpu().numpy()

    single_model_configs = create_model_configs(
        data, best_hidden_channels, best_num_heads, best['dropout']
    )

    # Store trained single models for final ensemble
    trained_single_models = {}

    for model_name, params in single_model_configs.items():
        print(f"\nTraining single {model_name} model...")
        model_class = MODEL_CLASSES[model_name]

        # Create and train single model
        single_model = model_class(**params)
        single_model = single_model.to(device)
        optimizer = torch.optim.Adam(single_model.parameters(), lr=best['lr'])
        criterion = FocalLoss(alpha=1, gamma=2)

        trainer = Trainer(single_model, data, device, optimizer, criterion)
        best_stats, training_history = trainer.fit(epochs=100)

        # Store training history
        training_histories[f'{model_name}_Single'] = training_history

        # Store trained model for ensemble
        trained_single_models[model_name] = single_model

        # Get predictions and probabilities
        single_model.eval()
        with torch.no_grad():
            out = single_model(data)
            probs = torch.exp(out).cpu().numpy()
            test_probs = probs[data.test_mask.cpu().numpy()]
            test_pred = np.argmax(test_probs, axis=1)

        # Calculate all metrics
        results_collector[f'{model_name}_Single'] = calculate_all_metrics(test_y_true, test_pred, test_probs)
        print(f"{model_name} single model evaluation completed")

    # ========================================================================
    # 6. Create and evaluate final ensemble model
    # ========================================================================
    print("\n6. Creating and evaluating final ensemble model...")

    final_ensemble = FinalEnsemble(trained_single_models)

    # Evaluate final ensemble model
    test_y_pred = final_ensemble.predict(data, device, test_only=True)

    # Get ensemble probabilities for all metrics calculation
    ensemble_probs = final_ensemble.predict_proba(data, device, test_only=True)

    # Ensure shapes match
    assert test_y_true.shape[0] == test_y_pred.shape[0], f"Shape mismatch: test_y_true {test_y_true.shape[0]}, test_y_pred {test_y_pred.shape[0]}"
    assert test_y_true.shape[0] == ensemble_probs.shape[0], f"Shape mismatch: test_y_true {test_y_true.shape[0]}, ensemble_probs {ensemble_probs.shape[0]}"

    # Calculate all metrics using ensemble predictions and probabilities
    results_collector['Final_Ensemble'] = calculate_all_metrics(test_y_true, test_y_pred, ensemble_probs)

    # Display final ensemble results
    print("\n" + "="*50)
    print("Final Ensemble Model Evaluation Results")
    print("="*50)
    ensemble_results = results_collector['Final_Ensemble']
    print(f"F1 Score: {ensemble_results['f1']:.4f}")
    print(f"Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"Precision: {ensemble_results['precision']:.4f}")
    print(f"Recall: {ensemble_results['recall']:.4f}")
    print(f"Macro Recall: {ensemble_results['macro_recall']:.4f}")
    print(f"Macro F1: {ensemble_results['macro_f1']:.4f}")
    print(f"AUC: {ensemble_results['auc']:.4f}")
    print(f"Macro AUC: {ensemble_results['macro_auc']:.4f}")
    print(f"G-Mean: {ensemble_results['gmean']:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_y_true, test_y_pred, target_names=['licit', 'illicit'], zero_division=0))

    # ========================================================================
    # 7. Generate visualizations and training curves
    # ========================================================================
    
    generate_standard_gnn_visualizations(results_collector, training_histories, test_y_true, test_y_pred)


    print("\n" + "="*60)
    print("Complete pipeline execution finished!")
    print("Results saved to results/ directory")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()
