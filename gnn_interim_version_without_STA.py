"""
gnn_interim_version.py

Advanced GNN framework version focused on blockchain anomaly detection
Includes: GCN, GAT, GIN, GraphSAGE (multi-hop neighbor information learning)
Ensemble + Bagging, Hyperopt (Bayesian Optimization)
Focal Loss for class imbalance, GAN-based data augmentation, PyTorch Geometric MLP
Isolation Forest Baseline for performance comparison

Execute all functions with one click, no command line arguments needed
Includes automatic baseline evaluation and performance comparison
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Hyperopt imports
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt

# Visualization tools
from visualization_tools import plot_confusion_matrix, plot_feature_visualization_2d, plot_training_curves, TrainingHistory

# Additional visualization imports for explain_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from torch_geometric.utils import to_networkx
import seaborn as sns

warnings.filterwarnings('ignore')

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
        # Convert inputs to probabilities
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # (N, C, H, W) -> (N, C, H*W)
            inputs = inputs.transpose(1, 2)    # (N, C, H*W) -> (N, H*W, C)
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # (N*H*W, C)

        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:                                                     ## 做多咗？
            self.convs.append(GCNConv(in_channels, out_channels)) ## 做多咗？
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, dropout=dropout, concat=False))
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # Use torch_geometric.nn.MLP for GINConv
        mlp_channels = [in_channels, hidden_channels, hidden_channels]
        self.convs.append(GINConv(MLP(mlp_channels, dropout=dropout)))
        for _ in range(num_layers - 2):
            mlp_channels = [hidden_channels, hidden_channels, hidden_channels]
            self.convs.append(GINConv(MLP(mlp_channels, dropout=dropout)))
        if num_layers > 1:
            mlp_channels = [hidden_channels, hidden_channels, out_channels]
            self.convs.append(GINConv(MLP(mlp_channels, dropout=dropout)))
        else:
            mlp_channels = [in_channels, hidden_channels, out_channels]
            self.convs.append(GINConv(MLP(mlp_channels, dropout=dropout)))
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

class Generator(nn.Module):
    """Simple GAN Generator for data augmentation"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
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


class Discriminator(nn.Module):
    """Simple GAN Discriminator for data augmentation"""
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
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


def augment_illegal_samples_with_gan(data, device, latent_dim=100, hidden_dim=128, num_epochs=100, augment_ratio=0.5):
    """Use GAN to generate additional illegal (class 1) samples"""
    print("Applying GAN-based data augmentation for illegal samples...")

    # Move data to CPU for processing to avoid CUDA issues
    data_cpu = data.cpu()

    # Extract illegal samples (class 1)
    illegal_mask = (data_cpu.y == 1)
    illegal_features = data_cpu.x[illegal_mask]
    num_illegal = illegal_features.size(0)
    num_to_generate = int(num_illegal * augment_ratio)

    if num_illegal == 0:
        print("No illegal samples found, skipping GAN augmentation")
        return data

    print(f"Found {num_illegal} illegal samples, will generate {num_to_generate} additional samples")

    # Initialize GAN models on CPU to avoid CUDA issues
    feature_dim = data.x.size(1)
    generator = Generator(latent_dim, hidden_dim, feature_dim)
    discriminator = Discriminator(feature_dim, hidden_dim)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training data for discriminator (on CPU)
    real_labels = torch.ones(num_illegal, 1)
    fake_labels = torch.zeros(num_illegal, 1)

    # Training loop
    for epoch in range(num_epochs):
        # Train Discriminator
        d_optimizer.zero_grad()

        # Real samples
        real_output = discriminator(illegal_features)
        d_loss_real = criterion(real_output, real_labels)

        # Fake samples
        z = torch.randn(num_illegal, latent_dim)
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
        z = torch.randn(num_to_generate, latent_dim)
        generated_features = generator(z)

        # Add small noise to make samples more diverse
        noise = torch.randn_like(generated_features) * 0.1
        generated_features = generated_features + noise

    # Create new data with augmented samples
    print(f"Original data shape: {data_cpu.x.shape}")
    print(f"Generated features shape: {generated_features.shape}")
    new_x = torch.cat([data_cpu.x, generated_features], dim=0)
    new_y = torch.cat([data_cpu.y, torch.ones(num_to_generate, dtype=torch.long)], dim=0)
    print(f"Augmented data shape: {new_x.shape}")

    # Create new masks (keep original train/val/test split, add new samples to training)
    new_train_mask = torch.cat([data_cpu.train_mask, torch.ones(num_to_generate, dtype=torch.bool)], dim=0)
    new_val_mask = torch.cat([data_cpu.val_mask, torch.zeros(num_to_generate, dtype=torch.bool)], dim=0)
    new_test_mask = torch.cat([data_cpu.test_mask, torch.zeros(num_to_generate, dtype=torch.bool)], dim=0)

    # Create augmented data object
    augmented_data = Data(
        x=new_x,
        y=new_y,
        edge_index=data_cpu.edge_index,  # Keep original edges
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
        try:
            # Convert anomaly scores to probability-like scores for AUC
            # Isolation Forest scores: negative = more anomalous
            # Convert to positive scores where higher = more likely illegal
            prob_scores = -anomaly_scores  # Flip sign so higher = more anomalous
            auc = roc_auc_score(y_true, prob_scores)
            macro_auc = auc  # For binary classification
        except:
            auc = float('nan')
            macro_auc = float('nan')

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
    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading data from: {dataset_dir}")
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')

    print(f"Full paths:")
    print(f"  classes: {classes_path}")
    print(f"  edgelist: {edgelist_path}")
    print(f"  features: {features_path}")
    print(f"Files found: classes={os.path.exists(classes_path)}, edgelist={os.path.exists(edgelist_path)}, features={os.path.exists(features_path)}")

    if not all(os.path.exists(p) for p in [classes_path, edgelist_path, features_path]):
        print("Dataset files not found in standard location. Searching other locations...")
        # Try to find dataset files in current directory or subdirectories
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            if 'elliptic_txs_classes.csv' in files and 'elliptic_txs_edgelist.csv' in files and 'elliptic_txs_features.csv' in files:
                dataset_dir = root
                print(f"Found dataset files in: {dataset_dir}")
                classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
                edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
                features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')
                break
        else:
            raise FileNotFoundError("Dataset files not found. Please ensure elliptic_txs_classes.csv, elliptic_txs_edgelist.csv, and elliptic_txs_features.csv are available.")
    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)
    features_df.rename(columns={0: 'txId'}, inplace=True)

    print(f"classes_df shape: {classes_df.shape}, columns: {classes_df.columns.tolist()}")
    print(f"features_df shape: {features_df.shape}, columns: {features_df.columns.tolist()[:10]}...")  # Show first 10 columns

    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    print(f"nodes_df shape after merge: {nodes_df.shape}")
    print(f"nodes_df columns: {nodes_df.columns.tolist()[:10]}...{nodes_df.columns.tolist()[-5:]}")  # Show first 10 and last 5

    feature_columns = nodes_df.columns[2:-1]  # Skip txId (0), timestep (1), and class (last)
    print(f"Using feature columns: {len(feature_columns)} columns")
    print(f"Feature column names: {feature_columns.tolist()[:5]}...{feature_columns.tolist()[-5:]}")

    features = nodes_df[feature_columns].values
    x = torch.tensor(features, dtype=torch.float)

    print(f"Loaded {x.size(0)} nodes with {x.size(1)} features")

    # Sanity check - Elliptic dataset should have ~165 features
    if x.size(1) != 165:
        print(f"WARNING: Expected 165 features but got {x.size(1)} features!")
        print(f"This might indicate incorrect dataset loading.")
        print(f"Check that you're using the correct Elliptic dataset files.")
        print(f"Expected: elliptic_txs_features.csv with 167 columns (1 txId + 1 timestep + 165 features)")
        print(f"Found: {len(nodes_df.columns)} total columns in merged dataframe")
        if x.size(1) < 10:
            print(f"First few feature values: {features[0][:10] if len(features) > 0 else 'No data'}")
    labels = nodes_df['class'].apply(lambda c: 1 if c == '2' else (0 if c == '1' else -1))
    y = torch.tensor(labels.values, dtype=torch.long)
    tx_ids = nodes_df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}
    source_indices = []
    target_indices = []
    for _, row in edgelist_df.iterrows():
        src = row['txId1'] if 'txId1' in edgelist_df.columns else row.iloc[0]
        tgt = row['txId2'] if 'txId2' in edgelist_df.columns else row.iloc[1]
        if src in tx_id_map and tgt in tx_id_map:
            source_indices.append(tx_id_map[src]); target_indices.append(tx_id_map[tgt])
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    timesteps = torch.tensor(nodes_df.iloc[:, 1].values, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps
    known_mask = y != -1
    data.train_mask = (timesteps < 35) & known_mask
    data.val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    data.test_mask = (timesteps >= 42) & known_mask
    return data


# -----------------------
# Ensemble Model with Bagging
# -----------------------

class BaggingEnsembleModel:
    """
    Ensemble learning framework with Bagging (maintain class imbalance ratio)
    """
    def __init__(self, model_class, model_params, n_estimators=5, voting='soft'):
        self.model_class = model_class
        self.model_params = model_params
        self.n_estimators = n_estimators
        self.voting = voting
        self.models = []
        self.bag_indices = []  # Record training indices for each bag

    def create_balanced_bags(self, data, n_bags=5):
        """Create bags that maintain class imbalance ratio"""
        train_idx = torch.where(data.train_mask)[0].cpu().numpy()
        train_labels = data.y[train_idx].cpu().numpy()

        # Calculate class ratios in original training set
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        label_ratios = counts / len(train_labels)

        bags = []
        for _ in range(n_bags):
            bag_indices = []
            for label, ratio in zip(unique_labels, label_ratios):
                label_indices = train_idx[train_labels == label]
                n_samples = max(1, int(len(label_indices) * ratio))
                # Bootstrap sampling (with replacement)
                sampled_indices = np.random.choice(label_indices, size=n_samples, replace=True)
                bag_indices.extend(sampled_indices)
            bags.append(np.array(bag_indices))

        return bags

    def fit(self, data, device, epochs=100, criterion=None):
        """Train ensemble model"""
        if criterion is None:
            criterion = nn.NLLLoss()

        # Create bags
        bags = self.create_balanced_bags(data, self.n_estimators)

        for i, bag_indices in enumerate(bags):
            print(f"Training bag {i+1}/{self.n_estimators}...")

            # Create sub-model
            model = self.model_class(**self.model_params)
            model = model.to(device)

            # Create sub-dataset
            bag_mask = torch.zeros_like(data.train_mask)
            bag_mask[bag_indices] = True

            # Train sub-model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out[bag_mask], data.y[bag_mask])
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    print(f"Bag {i+1}, Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

            self.models.append(model)
            self.bag_indices.append(bag_indices)

    def predict(self, data, device):
        """Make predictions"""
        all_predictions = []
        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(data)
                probs = torch.exp(output)
                preds = output.argmax(dim=1)
                all_predictions.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        if self.voting == 'soft':
            avg_probs = np.mean(all_probs, axis=0)
            ensemble_pred = np.argmax(avg_probs, axis=1)
        else:
            all_predictions = np.array(all_predictions)
            ensemble_pred = []
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                ensemble_pred.append(Counter(votes).most_common(1)[0][0])
            ensemble_pred = np.array(ensemble_pred)

        return ensemble_pred


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
    def evaluate(self, detailed=False):
        self.model.eval()
        out = self.model(self.data)
        pred = out.argmax(dim=1)
        val_loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask]).item()
        val_y_true = self.data.y[self.data.val_mask].cpu().numpy()
        val_y_pred = pred[self.data.val_mask].cpu().numpy()
        val_f1 = f1_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
        val_acc = accuracy_score(val_y_true, val_y_pred)

        # Test set evaluation
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
        try:
            test_auc = roc_auc_score(test_y_true, probs[test_mask_np][:, 1])
            test_macro_auc = test_auc  # For binary classification, macro AUC equals regular AUC
        except:
            test_auc = float('nan')
            test_macro_auc = float('nan')

        # G-Mean
        test_sensitivity = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
        test_specificity = recall_score(test_y_true, test_y_pred, pos_label=0, zero_division=0)
        test_gmean = np.sqrt(test_sensitivity * test_specificity) if test_sensitivity * test_specificity > 0 else 0

        return {
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'test_precision_illicit': test_precision_illicit,
            'test_recall_illicit': test_recall_illicit,
            'test_macro_recall': test_macro_recall,
            'test_macro_f1': test_macro_f1,
            'test_auc': test_auc,
            'test_macro_auc': test_macro_auc,
            'test_gmean': test_gmean
        }

    def fit(self, epochs=100, reduction_method='pca'): #extract_features_after=False, 
        best_val_loss = float('inf')
        best_stats = None

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            stats = self.evaluate()

            self.history.add_epoch(
                epoch=epoch + 1,  # Epoch starts from 1
                train_loss=train_loss,
                val_loss=stats['val_loss'],
                val_f1=stats['val_f1'],
                test_f1=stats['test_f1'],
                val_acc=stats['val_acc'],
                test_acc=stats['test_acc'],
                val_macro_recall=stats.get('test_macro_recall', 0),
                test_macro_recall=stats['test_macro_recall'],
                val_gmean=stats.get('test_gmean', 0),
                test_gmean=stats['test_gmean'],
                val_macro_f1=stats.get('test_macro_f1', 0),
                test_macro_f1=stats['test_macro_f1'],
                val_macro_auc=stats.get('test_macro_auc', 0),
                test_macro_auc=stats['test_macro_auc']
            )

            if stats['val_loss'] < best_val_loss:
                best_val_loss = stats['val_loss']
                best_stats = stats

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {stats['val_loss']:.4f}, Val F1: {stats['val_f1']:.4f}, "
                      f"Test F1: {stats['test_f1']:.4f}")

        # # Feature extraction and visualization
        # if extract_features_after:
        #     print("\nExtracting features and performing visualization...")
        #     extracted, reduced = perform_feature_extraction_and_reduction(
        #         self.model, self.data, self.device, reduction_method=reduction_method
        #     )

        return best_stats


class FinalEnsemble:
    """Final ensemble combining multiple model ensembles"""
    def __init__(self, models):
        self.models = models  # ensemble_models dict

    def predict(self, data, device, test_only=False):
        all_predictions = []
        all_probs = []

        for model_name, ensemble in self.models.items():
            pred = ensemble.predict(data, device)
            all_predictions.append(pred)

            # Calculate average probabilities (soft voting)
            probs_list = []
            for sub_model in ensemble.models:
                sub_model.eval()
                with torch.no_grad():
                    output = sub_model(data)
                    probs = torch.exp(output).cpu().numpy()
                    probs_list.append(probs)
            avg_probs = np.mean(probs_list, axis=0)
            all_probs.append(avg_probs)

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
        """Return averaged probabilities from all ensemble models"""
        all_probs = []

        for model_name, ensemble in self.models.items():
            # Calculate average probabilities (soft voting) for each model type
            probs_list = []
            for sub_model in ensemble.models:
                sub_model.eval()
                with torch.no_grad():
                    output = sub_model(data)
                    probs = torch.exp(output).cpu().numpy()
                    probs_list.append(probs)
            avg_probs = np.mean(probs_list, axis=0)
            all_probs.append(avg_probs)

        # Final soft voting across all model types
        final_avg_probs = np.mean(all_probs, axis=0)

        if test_only:
            # Return only test probabilities
            test_mask_np = data.test_mask.cpu().numpy()
            return final_avg_probs[test_mask_np]
        else:
            return final_avg_probs


# -----------------------
# Feature Extraction
# -----------------------

@torch.no_grad()
def extract_features(model, data, device, layer_idx=-1):
    model.eval()
    out = model(data, return_embed=True)
    if isinstance(out, tuple):
        _, embed = out
    else:
        embed = out
    if isinstance(embed, dict):
        embed = torch.cat([v for v in embed.values()], dim=1)
    return embed.cpu().numpy()


def reduce_dimension_pca(features, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    return reduced


def reduce_dimension_tsne(features, n_components=2, perplexity=30, n_iter=1000):
    if features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=0)
    reduced = tsne.fit_transform(features)
    return reduced


def perform_feature_extraction_and_reduction(model, data, device, reduction_method='pca', n_components=2, visualize=True, save_path=None):
    extracted = extract_features(model, data, device)
    if reduction_method.lower() == 'pca':
        reduced = reduce_dimension_pca(extracted, n_components)
        method_name = 'PCA'
    else:
        reduced = reduce_dimension_tsne(extracted, n_components)
        method_name = 't-SNE'
    if visualize and n_components == 2:
        plot_feature_visualization_2d(reduced, data.y, title=f"{type(model).__name__} Feature Visualization", method=method_name, save_path=save_path)
    return extracted, reduced



# -----------------------
# Hyperopt Objective Function
# -----------------------

def hyperopt_objective(params):
    """Hyperopt optimization objective function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load data
    data = load_elliptic_data()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Hyperopt returns actual values, not indices
    model_name = params['model']
    hidden_channels = params['hidden_channels']
    num_layers = params['num_layers']
    dropout = params['dropout']
    num_heads = params['num_heads']

    if model_name == 'GAT':
        model = GATModel(
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=2,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_name == 'GIN':
        model = GINModel(
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_name == 'GraphSAGE':
        model = GraphSAGEModel(
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        )
    else:  # GCN
        model = GCNModel(
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = FocalLoss(alpha=1, gamma=2)  # Use Focal Loss for class imbalance

    trainer = Trainer(model, data, device, optimizer, criterion)

    # Train
    best_stats = trainer.fit(epochs=50)  # Fewer training epochs to speed up optimization

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

    # Load data
    print("\n1. Loading Elliptic dataset...")
    data = load_elliptic_data()
    data = data.to(device)
    print(f"Data loading completed: {data.x.size(0)} nodes, {data.x.size(1)} features, {data.edge_index.size(1)} edges")

    # Train baseline model (Isolation Forest)
    print("\n1.2. Training Isolation Forest baseline model...")
    baseline_model = IsolationForestBaseline(
        n_estimators=100,
        contamination='auto',  # Auto-calculate from training data
        random_state=42
    )
    baseline_model.fit(data)
    baseline_results = baseline_model.evaluate(data)
    print("Baseline model evaluation completed")

    # Apply GAN-based data augmentation for illegal samples
    print("\n1.5. Applying GAN-based data augmentation...")
    data = augment_illegal_samples_with_gan(data, device, augment_ratio=0.3, num_epochs=50)  # Generate 30% more illegal samples
    print(f"Data augmentation completed: {data.x.size(0)} nodes total")

    # Hyperopt hyperparameter optimization
    print("\n2. Starting Hyperopt hyperparameter optimization...")

    # Define search space with reasonable minimum values
    space = {
        'model': hp.choice('model', ['GCN', 'GAT', 'GIN', 'GraphSAGE']),
        'hidden_channels': hp.choice('hidden_channels', [64, 128, 256]),  # Removed 32 to ensure sufficient capacity
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'num_layers': hp.choice('num_layers', [2, 3]),  # Limited to 2-3 layers to avoid dimension collapse
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'num_heads': hp.choice('num_heads', [4, 8])  # Limited to 4-8 heads
    }

    # Run optimization
    trials = Trials()
    # Skip hyperopt for debugging - use default parameters

    best = fmin(
        hyperopt_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=8,  # 30 evaluations
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Hyperopt returns actual values, not indices
    best_model = best['model']
    best_hidden_channels = best['hidden_channels']
    best_num_layers = best['num_layers']
    best_num_heads = best['num_heads']
    best_dropout = best['dropout']

    # Validate parameters to ensure they're reasonable
    if best_hidden_channels < 32:
        print(f"Warning: hidden_channels={best_hidden_channels} is very small, setting to 64")
        best_hidden_channels = 64
    if best_num_layers > 4:
        print(f"Warning: num_layers={best_num_layers} is large, setting to 3")
        best_num_layers = 3

    print("Best hyperparameters:")
    print(f"Model: {best_model}")
    print(f"Hidden channels: {best_hidden_channels}")
    print(f"Dropout: {best['dropout']:.3f}")
    print(f"Layers: {best_num_layers}")
    print(f"Learning rate: {best['lr']:.6f}")
    print(f"Num heads: {best_num_heads}")

    # Validate final parameters
    print(f"Final model config: {best_hidden_channels} hidden, {best_num_layers} layers, {best_num_heads} heads")

    # Train final model with best parameters
    print("\n3. Training final ensemble model with best parameters...")

    # Ensure minimum reasonable values
    best_hidden_channels = max(best_hidden_channels, 32)
    best_num_layers = min(max(best_num_layers, 2), 4)
    best_num_heads = min(max(best_num_heads, 4), 16)

    print(f"Using validated parameters: hidden_channels={best_hidden_channels}, num_layers={best_num_layers}, num_heads={best_num_heads}")

    # Create BaggingEnsemble combining GAT, GIN, GraphSAGE
    model_configs = {
        'GAT': {
            'in_channels': data.x.size(1),
            'hidden_channels': best_hidden_channels,
            'out_channels': 2,
            'num_heads': best_num_heads,
            'num_layers': best_num_layers,
            'dropout': best['dropout']
        },
        'GIN': {
            'in_channels': data.x.size(1),
            'hidden_channels': best_hidden_channels,
            'out_channels': 2,
            'num_layers': best_num_layers,
            'dropout': best['dropout']
        },
        'GraphSAGE': {
            'in_channels': data.x.size(1),
            'hidden_channels': best_hidden_channels,
            'out_channels': 2,
            'num_layers': best_num_layers,
            'dropout': best['dropout']
        }
    }

    # Debug: Check data dimensions before training
    print(f"Data dimensions before ensemble training: {data.x.size(0)} nodes, {data.x.size(1)} features")

    # Train four ensemble models
    ensemble_models = {}
    for model_name, params in model_configs.items():
        print(f"\nTraining {model_name} Bagging Ensemble...")
        print(f"Model params: in_channels={params['in_channels']}, hidden_channels={params['hidden_channels']}")
        if model_name == 'GAT':
            model_class = GATModel
        elif model_name == 'GIN':
            model_class = GINModel
        elif model_name == 'GraphSAGE':
            model_class = GraphSAGEModel


        ensemble = BaggingEnsembleModel(
            model_class=model_class,
            model_params=params,
            n_estimators=5,  # 5 bags
            voting='soft'
        )

        ensemble.fit(data, device, epochs=100)
        ensemble_models[model_name] = ensemble

    # Create final ensemble
    print("\n4. Creating final ensemble model (GAT + GIN + GraphSAGE)...")

    final_ensemble = FinalEnsemble(ensemble_models)

    # Define test labels for evaluation
    test_y_true = data.y[data.test_mask].cpu().numpy()

    # Train standalone GCN baseline model
    print("\n4.5. Training standalone GCN baseline model...")
    gcn_model = GCNModel(
        in_channels=data.x.size(1),
        hidden_channels=best_hidden_channels,
        out_channels=2,
        num_layers=best_num_layers,
        dropout=best_dropout
    )
    gcn_model = gcn_model.to(device)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=best['lr'])
    gcn_criterion = FocalLoss(alpha=1, gamma=2)

    gcn_trainer = Trainer(gcn_model, data, device, gcn_optimizer, gcn_criterion)
    gcn_best_stats = gcn_trainer.fit(epochs=100)  # Train GCN baseline for more epochs

    # Get GCN predictions and probabilities for test set
    gcn_model.eval()
    with torch.no_grad():
        gcn_out = gcn_model(data)
        gcn_probs = torch.exp(gcn_out).cpu().numpy()
        gcn_test_probs = gcn_probs[data.test_mask.cpu().numpy()]
        gcn_test_pred = np.argmax(gcn_test_probs, axis=1)

    # Calculate GCN baseline metrics
    gcn_f1 = f1_score(test_y_true, gcn_test_pred, average='binary', pos_label=1, zero_division=0)
    gcn_accuracy = accuracy_score(test_y_true, gcn_test_pred)
    gcn_precision = precision_score(test_y_true, gcn_test_pred, pos_label=1, zero_division=0)
    gcn_recall = recall_score(test_y_true, gcn_test_pred, pos_label=1, zero_division=0)
    gcn_macro_recall = recall_score(test_y_true, gcn_test_pred, average='macro', zero_division=0)
    gcn_macro_f1 = f1_score(test_y_true, gcn_test_pred, average='macro', zero_division=0)

    try:
        gcn_auc = roc_auc_score(test_y_true, gcn_test_probs[:, 1])
        gcn_macro_auc = gcn_auc
    except:
        gcn_auc = float('nan')
        gcn_macro_auc = float('nan')

    gcn_sensitivity = recall_score(test_y_true, gcn_test_pred, pos_label=1, zero_division=0)
    gcn_specificity = recall_score(test_y_true, gcn_test_pred, pos_label=0, zero_division=0)
    gcn_gmean = np.sqrt(gcn_sensitivity * gcn_specificity) if gcn_sensitivity * gcn_specificity > 0 else 0

    gcn_results = {
        'f1': gcn_f1, 'accuracy': gcn_accuracy, 'precision': gcn_precision,
        'recall': gcn_recall, 'macro_recall': gcn_macro_recall, 'macro_f1': gcn_macro_f1,
        'auc': gcn_auc, 'macro_auc': gcn_macro_auc, 'gmean': gcn_gmean
    }

    # Evaluate final model
    print("\n5. Evaluating final ensemble model...")

    # Use final ensemble model for prediction
    test_y_pred = final_ensemble.predict(data, device, test_only=True)

    # Get ensemble probabilities for all metrics calculation
    ensemble_probs = final_ensemble.predict_proba(data, device, test_only=True)

    # Debug: print shapes to verify
    print(f"test_y_true shape: {test_y_true.shape}")
    print(f"test_y_pred shape: {test_y_pred.shape}")
    print(f"ensemble_probs shape: {ensemble_probs.shape}")

    # Ensure shapes match
    assert test_y_true.shape[0] == test_y_pred.shape[0], f"Shape mismatch: test_y_true {test_y_true.shape[0]}, test_y_pred {test_y_pred.shape[0]}"
    assert test_y_true.shape[0] == ensemble_probs.shape[0], f"Shape mismatch: test_y_true {test_y_true.shape[0]}, ensemble_probs {ensemble_probs.shape[0]}"

    # Calculate all metrics using ensemble predictions and probabilities
    f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
    accuracy = accuracy_score(test_y_true, test_y_pred)
    precision = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
    recall = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
    macro_recall = recall_score(test_y_true, test_y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(test_y_true, test_y_pred, average='macro', zero_division=0)

    # AUC using integrated ensemble probabilities
    try:
        auc = roc_auc_score(test_y_true, ensemble_probs[:, 1])
        macro_auc = auc
    except:
        auc = float('nan')
        macro_auc = float('nan')

    # G-Mean using ensemble predictions
    sensitivity = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(test_y_true, test_y_pred, pos_label=0, zero_division=0)
    gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0

    print("\n" + "="*50)
    print("Final Ensemble Model Evaluation Results")
    print("="*50)
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"G-Mean: {gmean:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_y_true, test_y_pred, target_names=['Legal', 'Illegal'], zero_division=0))

    # Performance comparison with both baselines
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: Final Ensemble vs GCN Baseline vs IForest Baseline")
    print("="*80)

    # Create comparison table
    metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Macro Recall', 'Macro F1', 'AUC', 'Macro AUC', 'G-Mean']
    ensemble_scores = [f1, accuracy, precision, recall, macro_recall, macro_f1, auc, macro_auc, gmean]
    gcn_scores = [
        gcn_results['f1'], gcn_results['accuracy'], gcn_results['precision'],
        gcn_results['recall'], gcn_results['macro_recall'], gcn_results['macro_f1'],
        gcn_results['auc'], gcn_results['macro_auc'], gcn_results['gmean']
    ]
    iforest_scores = [
        baseline_results['f1'], baseline_results['accuracy'], baseline_results['precision'],
        baseline_results['recall'], baseline_results['macro_recall'], baseline_results['macro_f1'],
        baseline_results['auc'], baseline_results['macro_auc'], baseline_results['gmean']
    ]

    print(f"{'Metric':<15} {'Final Ensemble':>14} {'GCN Baseline':>12} {'vs GCN':>10} {'IForest':>10} {'vs IForest':>11}")
    print("-" * 90)

    for metric, ensemble_score, gcn_score, iforest_score in zip(metrics, ensemble_scores, gcn_scores, iforest_scores):
        # Format scores
        if not np.isnan(ensemble_score):
            ensemble_str = f"{ensemble_score:>14.4f}"
        else:
            ensemble_str = f"{'N/A':>14}"

        if not np.isnan(gcn_score):
            gcn_str = f"{gcn_score:>12.4f}"
        else:
            gcn_str = f"{'N/A':>12}"

        if not np.isnan(iforest_score):
            iforest_str = f"{iforest_score:>10.4f}"
        else:
            iforest_str = f"{'N/A':>10}"

        # Calculate improvements
        if not (np.isnan(ensemble_score) or np.isnan(gcn_score)):
            gcn_improvement = ((ensemble_score - gcn_score) / gcn_score) * 100
            gcn_imp_str = f"{gcn_improvement:+>9.1f}%"
        else:
            gcn_imp_str = f"{'N/A':>10}"

        if not (np.isnan(ensemble_score) or np.isnan(iforest_score)):
            iforest_improvement = ((ensemble_score - iforest_score) / iforest_score) * 100
            iforest_imp_str = f"{iforest_improvement:+>10.1f}%"
        else:
            iforest_imp_str = f"{'N/A':>11}"

        print(f"{metric:<15} {ensemble_str} {gcn_str} {gcn_imp_str} {iforest_str} {iforest_imp_str}")

    print("-" * 90)

    # Calculate average improvements
    gcn_avg_improvement = np.nanmean([
        ((e - g) / g) * 100 if not (np.isnan(e) or np.isnan(g)) else np.nan
        for e, g in zip(ensemble_scores, gcn_scores)
    ])

    iforest_avg_improvement = np.nanmean([
        ((e - i) / i) * 100 if not (np.isnan(e) or np.isnan(i)) else np.nan
        for e, i in zip(ensemble_scores, iforest_scores)
    ])

    if not np.isnan(gcn_avg_improvement):
        gcn_avg_str = f"{gcn_avg_improvement:+>9.1f}%"
    else:
        gcn_avg_str = f"{'N/A':>10}"

    if not np.isnan(iforest_avg_improvement):
        iforest_avg_str = f"{iforest_avg_improvement:+>10.1f}%"
    else:
        iforest_avg_str = f"{'N/A':>11}"

    print(f"{'Average':<15} {'':>14} {'':>12} {gcn_avg_str} {'':>10} {iforest_avg_str}")
    print("\n" + "="*60)

    # Plot confusion matrix
    print("\n6. Generating visualizations...")
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(test_y_true, test_y_pred, model_name='Final Ensemble Model',
                         save_path='results/confusion_matrix.png')


    print("\n" + "="*60)
    print("Complete pipeline execution finished!")
    print("Results saved to results/ directory")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()
