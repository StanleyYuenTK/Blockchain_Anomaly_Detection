# data preprocessing
## 0 done. catboosting (做完，但未理解，遲啲再解)
## 3 done. FocalLoss
## 1. GA - done / optuna - done
## 2. update dataset
## 4. MixHop power [0, 1, 2, 3], [0, 1, 2, 3, 4]
## 5. optimizer = torch.optim.Adam(model.parameters(), lr=lr)?
## 
## 異常節點的直接鄰居應判斷為高機率異常節點，如果異常節點的直接鄰居沒有直接鄰居是否可被視為必定為異常節點？
## blockchain dataset係 direction graph??
## generalization 同可解釋性intermitibility係同點？

# 2026/3/11 已更新optuna 所有model，已加入GA，但未test
# 2026/3/12 已加入GA，但已test
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
import torch_scatter
from torch_geometric.data import Data
from torch_geometric.utils import degree, get_ppr
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from community import community_louvain

# model
from sklearn.ensemble import IsolationForest

from catboost import CatBoostClassifier

# Optimization
import optuna
from optuna.samplers import TPESampler
import pygad

# visualiz
from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations

# GNN Models
import inspect
import GNNs
import load_dataset

## Focal Loss https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
from kornia.losses import FocalLoss

RANDOM_SEED = 24027277
n_trials=30

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


# ==============================================================================
# Genetic Algorithm
# https://pygad.readthedocs.io/en/latest/
# ==============================================================================

# 假設 X_val_meta 是你所有 GNN 預測結果組成的 DataFrame
def fitness_func(ga_instance, solution, solution_idx, X_val_meta, data, y_val):
    # solution 是 GA 產生的 [1, 0, 1...] 陣列
    selected_cols = [i for i, bit in enumerate(solution) if bit == 1]
    
    if len(selected_cols) == 0: return 0
    
    # 只選取部分 GNN 的預測作為特徵
    X_val_subset = X_val_meta[:, selected_cols]
    X_val_raw_and_meta = data.x[data.val_mask].cpu().numpy()
    X_subset = np.hstack([X_val_raw_and_meta, X_val_subset])
    
    # 訓練一個簡單的 CatBoost 作為評估 (為了速度，可以減少 iterations)
    clf = CatBoostClassifier(iterations=100, silent=True)
    clf.fit(X_subset, y_val)
    
    # 拿 Macro F1 score
    preds = clf.predict(X_subset)
    probs = clf.predict_proba(X_subset)[:, 1]

    c1_f1 = f1_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average='macro')
    auc = roc_auc_score(y_val, probs)
    weight_score = (0.4 * c1_f1) + (0.2 * macro_f1) + (0.4 * auc)
    return weight_score


# ==============================================================================
# TPE - Optuna
# ==============================================================================

def gnn_objective(trial, model_name, data):
    """Optuna objective function for GNN model optimization"""

    # Define search space
    in_channels = data.x.size(1)
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
    out_channels = 2
    dropout = trial.suggest_float('dropout', 0.1, 0.5, log=False)
    heads = trial.suggest_categorical('heads', [4, 8])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # Create model
    best_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'out_channels': out_channels,
        'dropout': dropout,
        'heads': heads,
        'lr': lr,
    }
    model = get_gnn(model_name, data, best_params)
    model = train_gnn(model, data, epochs=100, lr=lr)
    val_probs, val_preds, test_probs, test_preds = test_gnn(model, data)
    model_val_performance, model_test_performance = eval_gnn(model_name, data.y[data.val_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy(), val_probs, val_preds, test_probs, test_preds)
    ## macro f1 score, class 1 f1-score, auc
    return model_val_performance['macro f1-score'], model_val_performance['class 1 f1-score'], model_val_performance['auc']


def catboost_objective(trial, data, gnns_val_probs):
    """Optuna objective function for CatBoost model optimization"""
    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate
    X_val_raw_meta = np.hstack([data.x[data.val_mask].cpu().numpy(), X_val_meta])
    y_val = data.y[data.val_mask].cpu().numpy()
    print(f"X_val_meta shape: {X_val_meta.shape}")
    print(f"X_val_final shape: {X_val_raw_meta.shape}")
    X_tr, X_ev, y_tr, y_ev = train_test_split(X_val_raw_meta, y_val, test_size=0.25, random_state=RANDOM_SEED)

    # Define search space
    iterations = trial.suggest_int('iterations', 100, 500, step=100)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
    depth = trial.suggest_categorical('depth', [4, 6, 8, 10])
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
    cat = CatBoostClassifier(
        iterations,
        learning_rate,
        depth,
        l2_leaf_reg,
        loss_function="Logloss",
        eval_metric='AUC',
        task_type='GPU',
        devices='0',
        early_stopping_rounds=30,
        random_seed=RANDOM_SEED,
        auto_class_weights='Balanced',
        verbose=100,
    )

    cat.fit(X_tr, y_tr)

    test_preds = cat.predict(X_ev)
    test_probs = cat.predict_proba(X_ev)[:, 1]
    auc = eval_blending_catboost(y_ev, test_preds, test_probs)
    return auc


# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================

# Isolation Forest Baseline
def isolation_forest_baseline(data):
    ## isolation forest baseline skip validation set
    X_train = data.x[data.train_mask | data.val_mask].cpu().numpy()
    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    # Train Isolation Forest
    clf = IsolationForest(random_state=24027277, contamination=0.05)
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

    print(f"Isolation Forest baseline results:")
    for key, value in baseline_results.items():
        print(f"{key}: {value}")
    print("Baseline evaluation completed")

    return baseline_results

# GNN Model
def get_gnn(model_name, data, best_params):
    device = data.x.device
    best_params['in_channels'] = data.x.size(1)
    best_params['out_channels'] = 2
    # Initialize GNN model
    model_func = getattr(GNNs, model_name, None)
    model = model_func(best_params).to(device)
    return model


def train_gnn(model, data, epochs=100, lr=0.01):
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model


def test_gnn(model, data, threshold=0.5): 
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
       
        val_probs = probs[data.val_mask.cpu().numpy()]
        val_preds = (val_probs > threshold).astype(int)

        test_probs = probs[data.test_mask.cpu().numpy()]
        test_preds = (test_probs > threshold).astype(int)

    return val_probs, val_preds, test_probs, test_preds


def eval_gnn(model_name, y_val, y_test, val_probs, val_preds, test_probs, test_preds):
    # print(f"\n===== (Validate Set) for overfitting, ensemble =====")
    # print(classification_report(y_val, val_preds, zero_division=0))
    val_report = classification_report(y_val, val_preds, zero_division=0, output_dict=True)
    val_auc = roc_auc_score(y_val, val_probs)
    # print(f"Validate AUC: {auc:.4f}")
    model_val_performance = {
        'model': model_name,
        'class 1 precision': val_report['1']['precision'],
        'class 1 recall': val_report['1']['recall'],
        'class 1 f1-score': val_report['1']['f1-score'],
        'class 0 precision': val_report['0']['precision'],
        'class 0 recall': val_report['0']['recall'],
        'class 0 f1-score': val_report['0']['f1-score'],
        'macro precision': val_report['macro avg']['precision'],
        'macro recall': val_report['macro avg']['recall'],
        'macro f1-score': val_report['macro avg']['f1-score'],
        'accuracy': val_report['accuracy'],
        'auc': val_auc,
    }

    # print(f"\n===== (Test Set) for compare each model performance =====")
    # print(classification_report(y_test, test_preds, zero_division=0))
    test_report = classification_report(y_test, test_preds, zero_division=0, output_dict=True)
    test_auc = roc_auc_score(y_test, test_probs)
    # print(f"Test AUC: {auc:.4f}")
    model_test_performance = {
        'model': model_name,
        'class 1 precision': test_report['1']['precision'],
        'class 1 recall': test_report['1']['recall'],
        'class 1 f1-score': test_report['1']['f1-score'],
        'class 0 precision': test_report['0']['precision'],
        'class 0 recall': test_report['0']['recall'],
        'class 0 f1-score': test_report['0']['f1-score'],
        'macro precision': test_report['macro avg']['precision'],
        'macro recall': test_report['macro avg']['recall'],
        'macro f1-score': test_report['macro avg']['f1-score'],
        'accuracy': test_report['accuracy'],
        'auc': test_auc,
    }

    
    return model_val_performance, model_test_performance

    
def train_and_test_gnn(model_name, data, best_params=None):
    # Extract best parameters or use defaults
    # in_channels = best_params.get('in_channels', None)
    # hidden_channels = best_params.get('hidden_channels', 64)
    # out_channels = best_params.get('out_channels', 2)
    # num_layers = best_params.get('num_layers', 2)
    # dropout = best_params.get('dropout', 0.5)
    # heads = best_params.get('heads', 8)
    epochs = best_params.get('epochs', 100)
    lr = best_params.get('lr', 0.01)
    # {'hidden_channels': 32, 'num_layers': 4, 'dropout': 0.4062143730734663, 'heads': 4, 'lr': 0.0002074068439961685}.
    model = get_gnn(model_name, data, best_params)
    model = train_gnn(model, data, epochs, lr)
    return test_gnn(model, data)
     

# ==============================================================================
# Catboost Model - Blending
# ==============================================================================

def eval_blending_catboost(y_test, test_preds, test_probs):
     ## preformance metrics
    metrics = classification_report(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    print(metrics)
    print(f"Final CatBoost AUC: {auc:.4f}")
    return auc


def blending_catboost(data, gnns_val_probs, gnns_test_probs, best_params=None):
    ## process meta data
    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate
    X_test_meta = np.hstack(gnns_test_probs)
    print(f"X_val_meta shape: {X_val_meta.shape}, X_test_meta shape: {X_test_meta.shape}")

    X_val_raw_meta = np.hstack([data.x[data.val_mask].cpu().numpy(), X_val_meta])
    X_test_raw_meta = np.hstack([data.x[data.test_mask].cpu().numpy(), X_test_meta])
    print(f"X_val_final shape: {X_val_raw_meta.shape}, X_test_final shape: {X_test_raw_meta.shape}")

    # Extract best parameters or use defaults
    iterations = best_params.get('iterations', 500)
    learning_rate = best_params.get('learning_rate', 0.05)
    depth = best_params.get('depth', 4)
    l2_leaf_reg = best_params.get('l2_leaf_reg', 3.0)

    ## create catboost model and train and predict
    cat = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        auto_class_weights='Balanced', # 讓 CatBoost 自己處理不平衡，不影響 GNN 訓練
        verbose=100,
        eval_metric='AUC'
    )

    # fit val data, 
    # predict test data
    y_val = data.y[data.val_mask].cpu().numpy()
    cat.fit(X_val_raw_meta, y_val)

    test_preds = cat.predict(X_test_raw_meta)
    test_probs = cat.predict_proba(X_test_raw_meta)[:, 1]

    return test_preds, test_probs


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
    
    ## 暫時comment #######################################################
    print("Louvain features...")
    louvain_features, partition = get_louvain_features(elliptic_data.edge_index.cpu(), elliptic_data.x.size(0), labels=elliptic_data.y, train_mask=elliptic_data.train_mask)
    louvain_features = louvain_features.to(device)

    print("\nAdding pagerank, degree, louvain features to raw dataset...")
    elliptic_data.x = torch.cat([elliptic_data.x, pagerank_features, degree_features, louvain_features], dim=1)
    # elliptic_data.x = torch.cat([elliptic_data.x, pagerank_features, degree_features], dim=1)
    print(f"Total features: {elliptic_data.x.size(1)} dimensions")
    
    # Apply StandardScaler to normalize all features
    print("\nStandardScaler...")
    scaler = StandardScaler()
    x_numpy = elliptic_data.x.cpu().numpy()  # Convert to numpy
    scaler.fit(x_numpy[elliptic_data.train_mask.cpu().numpy()]) # 記住咗 Train Set 的 Mean 同 Std

    x_scaled = scaler.transform(x_numpy)  # Fit and transform
    elliptic_data.x = torch.tensor(x_scaled, dtype=torch.float).to(elliptic_data.x.device)  # Convert back to tensor
    print(f"StandardScaler done...\nTotal features: {elliptic_data.x.size(1)} dimensions")
    

    # ========================================================================
    # 3. Train Isolation Forest baseline - done - 暫時comment for train GNN model
    # ========================================================================
     ## 暫時comment #######################################################
    print("\n3. Training Isolation Forest baseline...")
    baseline_results = isolation_forest_baseline(elliptic_data)
    
    # ========================================================================
    # 4. Train GNN
    # 4.1. TPE 優化參數 
    # 4.2. train GNN
    # ========================================================================
    # 4.1. TPE - Optuna - done
    # ========================================================================
    print("\n4.1 TPE - Optuna...")
    results = {}
    gnn_models_list = inspect.getmembers(GNNs, inspect.isfunction)
    print(type(gnn_models_list))

    for model_name, _ in gnn_models_list:
        study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
        study.optimize(lambda trial: gnn_objective(trial, model_name, elliptic_data), n_trials=n_trials)
        w_m_f1, w2_f1, w3_auc = 0.4, 0.2, 0.4
        best_trials = max(study.best_trials, key=lambda t: w_m_f1 * t.values[0] + w2_f1 * t.values[1] + w3_auc * t.values[2])
        results[model_name] = {
            "macro_f1": best_trials.values[0],
            "c1_f1": best_trials.values[1],
            "auc": best_trials.values[2],
            "best_params": best_trials.params,
        }
        print(model_name, "="*60, "\n", results[model_name])

    # print(f"Best hyperparameters - {model_name}: {study.best_params}")
    # print(f"Best score - auc?: {study.best_value}")

    # ========================================================================
    # 4. Train GCN model
    # gnn_models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII',
    #     'MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN',
    #     'MixHop_GCN_K3', 'MixHop_GAT_K3', 'MixHop_GraphSAGE_K3', 'MixHop_GIN_K3',
    #     'MixHop_GCN_K4', 'MixHop_GAT_K4', 'MixHop_GraphSAGE_K4', 'MixHop_GIN_K4',
    # ]
    # ========================================================================
    print("\n4.2. Training GNNs baseline...")
    y_val = elliptic_data.y[elliptic_data.val_mask].cpu().numpy()
    y_test = elliptic_data.y[elliptic_data.test_mask].cpu().numpy()
    gnns_val_probs, gnns_test_probs = [], []
    models_val_performance, models_test_performance = [], []

    ## get GNN ZOO all model
    gnn_models_list = inspect.getmembers(GNNs, inspect.isfunction)
    # gnn_models_list.extend([('GCN', 0), ('GAT', 0), ('GraphSAGE', 0), ('GIN', 0)])
    print('gnn models length:', len(gnn_models_list))
    for model_name, _ in gnn_models_list:
        print(f"\n--- Training {model_name} ---")
        best_params = results.get(model_name, {}).get("best_params", {})
        gnn_val_probs, gnn_val_preds, gnn_test_probs, gnn_test_preds = train_and_test_gnn(model_name, elliptic_data, best_params)
        gnns_val_probs.append(gnn_val_probs.reshape(-1, 1))
        gnns_test_probs.append(gnn_test_probs.reshape(-1, 1))
        model_val_performance, model_test_performance = eval_gnn(model_name, y_val, y_test, gnn_val_probs, gnn_val_preds, gnn_test_probs, gnn_test_preds)
        models_val_performance.append(model_val_performance)
        models_test_performance.append(model_test_performance)

    # 1. 設置顯示所有 Column (唔好用 ... 隱藏)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    # 4. 限制小數點位數，令表格更對齊
    # pd.set_option('display.precision', 4)
    df_val_results = pd.DataFrame(models_val_performance)
    df_test_results = pd.DataFrame(models_test_performance)
    print("\n", df_val_results, "\n", "="*60, "\n", df_test_results)


    # ========================================================================
    # 5. training CatBoost
    # 5.1. GA 揀 model
    # 5.2 TPE - Optuna 優化catboost參數
    # 5.3. Blending CatBoost
    # ========================================================================
    # 5.1. GA
    # ========================================================================
    # 初始化 GA (11 個模型，所以基因長度是 11)
    print("\n5.1. GA Crossover model...")
    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate, GNN models gnn_val_probs
    bound_fitness = lambda ga_instance, solution, solution_idx: fitness_func(ga_instance, solution, solution_idx, X_val_meta, elliptic_data, y_val)
    # partial(fitness_func, X_data=X_val_raw_meta, y_true=y_val)
    print('gnn models length:', len(gnn_models_list))
    ga_instance = pygad.GA(num_generations=50, 
                    num_parents_mating=5, 
                    fitness_func=bound_fitness,
                    sol_per_pop=20, 
                    num_genes=len(gnn_models_list),
                    gene_space=[0, 1]) # 二進位搜尋
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    

    # ========================================================================
    # 5.2. TPE - Optuna - CatBoost
    # ========================================================================
    print(f"\n5.2 TPE - Optuna - CatBoost with \nGA selection: {solution}...")
    # 根據 GA 的 0/1 結果篩選 Probabilities
    selected_indices = [i for i, bit in enumerate(solution) if bit == 1]
    filtered_val_probs = [gnns_val_probs[i] for i in selected_indices]
    filtered_test_probs = [gnns_test_probs[i] for i in selected_indices]
    # 使用篩選後的 Probs 進行 Blending
    print(f"Using {len(selected_indices)} models selected by GA for final Blending.")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: catboost_objective(trial, elliptic_data, filtered_val_probs), n_trials=n_trials)
    cat_best_params = study.best_params
    cat_best_value = study.best_value
    print(f"Best hyperparameters - CatBoost: {cat_best_params}")
    print(f"Best score - auc?: {cat_best_value}")
    
    # study = optuna.create_study(direction='maximize')
    # study.optimize(lambda trial: catboost_objective(trial, elliptic_data, gnns_val_probs, gnns_test_probs), n_trials=n_trials)
    # print(f"Best hyperparameters - CatBoost: {study.best_params}")
    # print(f"Best score - auc?: {study.best_value}")


    # ========================================================================
    # 5.3. softmax - Ensemble Model or blending ensemble model
    # Blending, Soft Voting, Bagging, stacking
    # 改用blending，因為blending比起stacking易做，不需要oof，
    # 因為blockchain dataset有節點和邊，不會因為time split，又不會因為節點的邊造成有data leakage問題，
    # 另外又有資料比例問題。
    # ========================================================================
    print("\n5.3 Blending CatBoost...")
    cat_test_preds, cat_test_probs = blending_catboost(elliptic_data, filtered_val_probs, filtered_test_probs, cat_best_params)
    # cat_test_preds, cat_test_probs = blending_catboost(elliptic_data, gnns_val_probs, gnns_test_probs)
    eval_blending_catboost(y_test, cat_test_preds, cat_test_probs)

    

if __name__ == "__main__":
    main()