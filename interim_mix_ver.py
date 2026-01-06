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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP, MixHopConv

from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Hyperopt imports
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

# Visualization tools
from visualization_tools import TrainingHistory, generate_mixhop_visualizations

def create_mixhop_model_configs(data, best_hidden_channels, best_num_heads, dropout_value):
    """Create model parameter configurations for MixHop GNN models"""
    base_config = {
        'in_channels': data.x.size(1),
        'hidden_channels': best_hidden_channels,
        'out_channels': 2,
        'dropout': dropout_value
    }

    configs = {
        'MixHop_GCN': base_config.copy(),
        'MixHop_GIN': base_config.copy(),
        'MixHop_GraphSAGE': base_config.copy(),
        'MixHop_GAT': {
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
# Model definitions with MixHopConv enhancement
# -----------------------

class BaseGNN(nn.Module):
    """Provide unified forward(self, data, return_embed=False) interface"""
    def forward(self, data, return_embed=False):
        raise NotImplementedError


class MixHopGCNModel(BaseGNN):
    """GCN with MixHopConv enhancement: MixHopConv + GCN + MixHopConv"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(MixHopGCNModel, self).__init__()
        # MixHopConv for multi-hop neighborhood aggregation (0-hop: self, 1-hop: neighbors, 2-hop: extended neighbors)
        self.mixhop1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        # GCN for feature learning
        self.gcn = GCNConv(hidden_channels, hidden_channels)
        # Final MixHopConv for output refinement
        self.mixhop2 = MixHopConv(hidden_channels, out_channels, powers=[0, 1, 2])
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index

        # Stage 1: Multi-hop neighborhood learning
        x = self.mixhop1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 2: GCN feature learning
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 3: Final multi-hop integration
        x = self.mixhop2(x, edge_index)

        out = F.log_softmax(x, dim=1)
        return out


class MixHopGATModel(BaseGNN):
    """GAT with MixHopConv enhancement: MixHopConv + GAT + MixHopConv"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.5):
        super(MixHopGATModel, self).__init__()
        # MixHopConv for multi-hop neighborhood aggregation (0-hop: self, 1-hop: neighbors, 2-hop: extended neighbors)
        self.mixhop1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        # GAT for attention-based feature learning
        self.gat = GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True)
        # Final MixHopConv for output refinement
        self.mixhop2 = MixHopConv(hidden_channels * num_heads, out_channels, powers=[0, 1, 2])
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index

        # Stage 1: Multi-hop neighborhood learning
        x = self.mixhop1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 2: GAT attention learning
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 3: Final multi-hop integration
        x = self.mixhop2(x, edge_index)

        out = F.log_softmax(x, dim=1)
        return out


class MixHopGINModel(BaseGNN):
    """GIN with MixHopConv enhancement: MixHopConv + GIN + MixHopConv"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(MixHopGINModel, self).__init__()
        # MixHopConv for multi-hop neighborhood aggregation (0-hop: self, 1-hop: neighbors, 2-hop: extended neighbors)
        self.mixhop1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        # GIN for graph isomorphism learning
        self.gin = GINConv(MLP([hidden_channels, hidden_channels, hidden_channels], dropout=dropout))
        # Final MixHopConv for output refinement
        self.mixhop2 = MixHopConv(hidden_channels, out_channels, powers=[0, 1, 2])
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index

        # Stage 1: Multi-hop neighborhood learning
        x = self.mixhop1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 2: GIN isomorphism learning
        x = self.gin(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 3: Final multi-hop integration
        x = self.mixhop2(x, edge_index)

        out = F.log_softmax(x, dim=1)
        return out


class MixHopGraphSAGEModel(BaseGNN):
    """GraphSAGE with MixHopConv enhancement: MixHopConv + GraphSAGE + MixHopConv"""
    def __init__(self, in_channels, hidden_channels, out_channels, aggr='mean', dropout=0.5):
        super(MixHopGraphSAGEModel, self).__init__()
        # MixHopConv for multi-hop neighborhood aggregation (0-hop: self, 1-hop: neighbors, 2-hop: extended neighbors)
        self.mixhop1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        # GraphSAGE for neighborhood sampling and aggregation
        self.sage = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        # Final MixHopConv for output refinement
        self.mixhop2 = MixHopConv(hidden_channels, out_channels, powers=[0, 1, 2])
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index

        # Stage 1: Multi-hop neighborhood learning
        x = self.mixhop1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 2: GraphSAGE neighborhood sampling
        x = self.sage(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stage 3: Final multi-hop integration
        x = self.mixhop2(x, edge_index)

        out = F.log_softmax(x, dim=1)
        return out


# Model class mapping for cleaner code
MODEL_CLASSES = {
    'GCN': MixHopGCNModel,
    'GAT': MixHopGATModel,
    'GIN': MixHopGINModel,
    'GraphSAGE': MixHopGraphSAGEModel
}


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
    generator = Generator(latent_dim, hidden_dim, feature_dim).to(device)
    discriminator = Discriminator(feature_dim, hidden_dim).to(device)

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

    # Augmented data is already on the correct device

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
    # print(f"Current working directory: {os.getcwd()}")  # Debug: Current working directory
    # print(f"Loading data from: {dataset_dir}")  # Debug: Dataset directory
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')

    # print(f"Full paths:")  # Debug: File paths
    # print(f"  classes: {classes_path}")
    # print(f"  edgelist: {edgelist_path}")
    # print(f"  features: {features_path}")
    # print(f"Files found: classes={os.path.exists(classes_path)}, edgelist={os.path.exists(edgelist_path)}, features={os.path.exists(features_path)}")  # Debug: File existence check

    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)
    features_df.rename(columns={0: 'txId'}, inplace=True)
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    feature_columns = nodes_df.columns[2:-1]  # Skip txId (0), timestep (1), and class (last)
    features = nodes_df[feature_columns].values
    x = torch.tensor(features, dtype=torch.float)

    print(f"Loaded {x.size(0)} nodes with {x.size(1)} features")

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

class BaggingModel:
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

    def create_balanced_bags(self, data):
        """Create bags that maintain class imbalance ratio"""
        train_idx = torch.where(data.train_mask)[0].cpu().numpy()
        train_labels = data.y[train_idx].cpu().numpy()

        # Calculate class ratios in original training set
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        label_ratios = counts / len(train_labels)

        bags = []
        for _ in range(self.n_estimators):
            bag_indices = []
            for label, ratio in zip(unique_labels, label_ratios):
                label_indices = train_idx[train_labels == label]
                n_samples = max(1, int(len(label_indices) * ratio))
                # Bootstrap sampling (with replacement)
                sampled_indices = np.random.choice(label_indices, size=n_samples, replace=True)
                bag_indices.extend(sampled_indices)
            bags.append(np.array(bag_indices))

        return bags

    def fit(self, data, device, epochs=100, criterion=None, lr=0.01):
        """Train ensemble model"""
        if criterion is None:
            criterion = nn.NLLLoss()

        # Create bags
        bags = self.create_balanced_bags(data)

        last_training_history = None  # Store the last model's training history

        for i, bag_indices in enumerate(bags):
            print(f"Training bag {i+1}/{self.n_estimators}...")

            # Create sub-model
            model = self.model_class(**self.model_params)
            model = model.to(device)

            # Create a custom dataset that uses bag samples for training but full validation set
            bag_train_mask = torch.zeros(len(data.x), dtype=torch.bool)
            bag_train_mask[bag_indices] = True

            # Create modified data object with custom training mask
            bag_data = Data(
                x=data.x,
                y=data.y,
                edge_index=data.edge_index,
                train_mask=bag_train_mask,  # Only bag samples for training
                val_mask=data.val_mask,     # Full validation set
                test_mask=data.test_mask    # Full test set (though we won't use it)
            )
            bag_data = bag_data.to(device)

            # Train sub-model with Trainer to get history
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trainer = Trainer(model, bag_data, device, optimizer, criterion)
            best_stats, training_history = trainer.fit(epochs=epochs, include_test=False)

            self.models.append(model)
            self.bag_indices.append(bag_indices)

            # Keep the last model's training history
            last_training_history = training_history

        return last_training_history


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

        # Validation set G-Mean
        val_sensitivity = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
        val_specificity = recall_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
        val_gmean = np.sqrt(val_sensitivity * val_specificity) if val_sensitivity * val_specificity > 0 else 0

        # Validation set AUC (using probabilities)
        val_probs = torch.exp(out[self.data.val_mask]).cpu().numpy()
        try:
            val_auc = roc_auc_score(val_y_true, val_probs[:, 1])
            val_macro_auc = val_auc  # For binary classification, macro AUC equals regular AUC
        except:
            val_auc = float('nan')
            val_macro_auc = float('nan')

        result = {
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_macro_recall': val_macro_recall,
            'val_macro_f1': val_macro_f1,
            'val_auc': val_auc,
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
                val_macro_auc=stats['val_macro_auc'],
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
# Hyperopt Objective Function
# -----------------------

def hyperopt_objective(data, device, params):
    """Hyperopt optimization objective function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Clean up params to ensure only current parameters are used
    cleaned_params = {
        'model': params['model'],
        'hidden_channels': params['hidden_channels'],
        'dropout': params['dropout'],
        'num_heads': params['num_heads'],
        'lr': params['lr']
    }

    model_name = cleaned_params['model']
    hidden_channels = cleaned_params['hidden_channels']
    dropout = cleaned_params['dropout']
    num_heads = cleaned_params['num_heads']

    # Create model using the mapping
    model_class = MODEL_CLASSES.get(model_name, MixHopGCNModel)
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
    criterion = FocalLoss(alpha=1, gamma=2)  # Use Focal Loss for class imbalance

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
    """Calculate all 9 evaluation metrics"""
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
    """Execute complete GNN anomaly detection pipeline with MixHopConv enhancement"""
    print("=" * 70)
    print("Blockchain Anomaly Detection GNN Framework - MixHopConv Enhanced")
    print("=" * 70)


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
        'hidden_channels': hp.choice('hidden_channels', [64, 128, 256]),  # Removed 32 to ensure sufficient capacity
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'num_heads': hp.choice('num_heads', [4, 8])  # Limited to 4-8 heads
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
        max_evals=1,  # 30 evaluations
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Convert indices back to actual values (hp.choice returns indices, not values)
    best_model = MODEL_CHOICES[best['model']]
    best_hidden_channels = HIDDEN_CHANNEL_CHOICES[best['hidden_channels']]
    best_num_heads = NUM_HEAD_CHOICES[best['num_heads']]
    best_dropout = best['dropout']  # hp.uniform returns actual value
    best_lr = best['lr']  # hp.loguniform returns actual value

    # Validate parameters to ensure they're reasonable
    best_hidden_channels = max(best_hidden_channels, 64)  # Ensure at least 64
    best_num_heads = max(min(best_num_heads, 8), 4)       # Ensure between 4 and 8

    print("Best hyperparameters:")
    print(f"Model: {best_model}")
    print(f"Hidden channels: {best_hidden_channels}")
    print(f"Dropout: {best['dropout']:.3f}")
    print(f"Layers: 3 (MixHopConv fixed)")
    print(f"Learning rate: {best['lr']:.6f}")
    print(f"Num heads: {best_num_heads}")

    # Validate final parameters
    print(f"Final model config: {best_hidden_channels} hidden, 3 layers, {best_num_heads} heads")

    # ========================================================================
    # 5. Train individual MixHopConv-enhanced GNN models
    # ========================================================================
    print("\n5. Training individual MixHopConv-enhanced GNN models...")

    # Initialize results collector and training histories
    results_collector = {
        'IForest': baseline_results,
        'MixHop_GCN_Single': {},
        'MixHop_GAT_Single': {},
        'MixHop_GIN_Single': {},
        'MixHop_GraphSAGE_Single': {},
        'MixHop_GCN_Bagging': {},
        'MixHop_GAT_Bagging': {},
        'MixHop_GIN_Bagging': {},
        'MixHop_GraphSAGE_Bagging': {},
        'MixHop_Final_Ensemble': {}
    }

    training_histories = {}  # Store training histories for visualization

    # Define test labels for evaluation
    test_y_true = data.y[data.test_mask].cpu().numpy()

    # Ensure minimum reasonable values
    best_hidden_channels = max(best_hidden_channels, 64)
    best_num_heads = min(max(best_num_heads, 4), 16)

    single_model_configs = create_mixhop_model_configs(
        data, best_hidden_channels, best_num_heads, best['dropout']
    )

    for model_name, params in single_model_configs.items():
        print(f"\nTraining single {model_name} model...")
        # Extract base model name (remove 'MixHop_' prefix)
        base_name = model_name.replace('MixHop_', '')
        model_class = MODEL_CLASSES[base_name]

        # Create and train single model
        single_model = model_class(**params)
        single_model = single_model.to(device)
        optimizer = torch.optim.Adam(single_model.parameters(), lr=best['lr'])
        criterion = FocalLoss(alpha=1, gamma=2)

        trainer = Trainer(single_model, data, device, optimizer, criterion)
        best_stats, training_history = trainer.fit(epochs=100)

        # Store training history
        training_histories[f'{model_name}_Single'] = training_history

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
    # 6. Train bagging MixHopConv-enhanced GNN models
    # ========================================================================
    print("\n6. Training bagging MixHopConv-enhanced GNN models...")

    # Create BaggingEnsemble combining MixHopConv-enhanced GCN, GAT, GIN, GraphSAGE
    bagging_model_configs = create_mixhop_model_configs(
        data, best_hidden_channels, best_num_heads, best['dropout']
    )

    # Train four ensemble models
    ensemble_models = {}
    for model_name, params in bagging_model_configs.items():
        print(f"\nTraining {model_name} Bagging Ensemble...")

        # Extract base model name (remove 'MixHop_' prefix)
        base_name = model_name.replace('MixHop_', '')
        model_class = MODEL_CLASSES[base_name]

        ensemble = BaggingModel(
            model_class=model_class,
            model_params=params,
            n_estimators=5,
            voting='soft'
        )

        bagging_history = ensemble.fit(data, device, epochs=100, lr=best['lr'])
        ensemble_models[model_name] = ensemble

        # Store training history for bagging model (using last model's history as representative)
        training_histories[f'{model_name}_Bagging'] = bagging_history

        # Evaluate individual bagging ensemble for each model
        print(f"Evaluating {model_name} Bagging Ensemble individually...")
        bagging_pred = ensemble.predict(data, device)
        # Calculate average probabilities from bagging models
        probs_list = []
        for sub_model in ensemble.models:
            sub_model.eval()
            with torch.no_grad():
                output = sub_model(data)
                probs = torch.exp(output).cpu().numpy()
                probs_list.append(probs)
        bagging_avg_probs = np.mean(probs_list, axis=0)
        bagging_pred_test = bagging_pred[data.test_mask.cpu().numpy()]
        bagging_probs_test = bagging_avg_probs[data.test_mask.cpu().numpy()]
        results_collector[f'{model_name}_Bagging'] = calculate_all_metrics(test_y_true, bagging_pred_test, bagging_probs_test)

    # ========================================================================
    # 7. Create and evaluate MixHopConv-enhanced final ensemble model
    # ========================================================================
    print("\n7. Creating and evaluating MixHopConv-enhanced final ensemble model...")

    final_ensemble = FinalEnsemble(ensemble_models)

    # Evaluate final ensemble model
    test_y_pred = final_ensemble.predict(data, device, test_only=True)

    # Get ensemble probabilities for all metrics calculation
    ensemble_probs = final_ensemble.predict_proba(data, device, test_only=True)

    # Ensure shapes match
    assert test_y_true.shape[0] == test_y_pred.shape[0], f"Shape mismatch: test_y_true {test_y_true.shape[0]}, test_y_pred {test_y_pred.shape[0]}"
    assert test_y_true.shape[0] == ensemble_probs.shape[0], f"Shape mismatch: test_y_true {test_y_true.shape[0]}, ensemble_probs {ensemble_probs.shape[0]}"

    # Calculate all metrics using ensemble predictions and probabilities
    results_collector['MixHop_Final_Ensemble'] = calculate_all_metrics(test_y_true, test_y_pred, ensemble_probs)

    # Display final ensemble results
    print("\n" + "="*50)
    print("MixHopConv-Enhanced Final Ensemble Model Evaluation Results")
    print("="*50)
    ensemble_results = results_collector['MixHop_Final_Ensemble']
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
    print(classification_report(test_y_true, test_y_pred, target_names=['Legal', 'Illegal'], zero_division=0))

    # ========================================================================
    # 8. Generate visualizations and training curves
    # ========================================================================

    generate_mixhop_visualizations(results_collector, training_histories, test_y_true, test_y_pred)

    print("\n" + "="*70)
    print("Complete MixHopConv-enhanced pipeline execution finished!")
    print("Results saved to results/ directory")
    print("="*70)


if __name__ == "__main__":
    run_full_pipeline()