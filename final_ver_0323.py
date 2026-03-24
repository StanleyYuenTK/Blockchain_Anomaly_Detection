# data preprocessing
## 0 done. catboosting (做完，但未理解，遲啲再解)
## 3 done. FocalLoss
## 1. done - GA - done / optuna - done
## 2. update dataset - ethereum, 
## 4. MixHop power [0, 1, 2, 3], [0, 1, 2, 3, 4]
## 5. optimizer = torch.optim.Adam(model.parameters(), lr=lr)?
## 6. 圖 (dataset imbalance, model performance)
##    6.1. Precision 被模型預測為該類別的樣本中，真正屬於該類別的比例。高 Precision 代表模型不亂抓，誤報（False Positive）少。
##    6.2. Recall 在所有實際屬於該類別的樣本中，被模型正確預測出來的比例。高 Recall 代表模型「不漏抓」，漏報（False Negative）少。
##    6.3. F1-score Precision 與 Recall 的調和平均數
##    6.4. Macro 當處理多類別或類別不平衡（Imbalanced Data）時，這些指標能反映模型對所有類別的平均照顧程度。Macro 系列指標對小類別（少數類）非常敏感。即便大類別預測完美，只要小類別表現差，Macro 指標就會大幅下降。這在區塊鏈異常檢測中極其重要。
##    6.5. Accuracy 全體樣本中預測正確（包含正負樣本）的比例。在數據極度不平衡時（例如 99% 的交易是正常的），即使模型將所有樣本都預測為正常，Accuracy 仍高達 99%，但此時模型毫無偵測能力。
##    6.6. AUC 模型有多大的機率能正確判斷出「異常那個比正常那個更可疑」1.0：完美模型。0.5：隨機猜測。< 0.5：模型反向預測了（比亂猜還慘）。


## 異常節點的直接鄰居應判斷為高機率異常節點，如果異常節點的直接鄰居沒有直接鄰居是否可被視為必定為異常節點？
## blockchain dataset係 direction graph??
## generalization 同可解釋性intermitibility係同點？

# 2026/3/11 已更新optuna 所有model，已加入GA，但未test
# 2026/3/12 已加入GA，已test
# 2026/3/23 修改為full batch
import gc
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
import torch_scatter
from torch_geometric.data import Data
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, confusion_matrix
# from torch_geometric.utils import degree, get_ppr
# from sklearn.preprocessing import StandardScaler
# from community import community_louvain
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.loader import NeighborLoader, ImbalancedSampler

# model
from sklearn.ensemble import IsolationForest

from catboost import CatBoostClassifier

# Optimization
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis
import pygad
import json
import copy

# visualiz
from visualization_tools import TrainingHistory, generate_standard_gnn_visualizations, plot_model_comparison

# GNN Models
import inspect
import gnn_zoo
import dataset_zoo

## Focal Loss https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
from kornia.losses import FocalLoss
import argparse

RANDOM_SEED = 24027277
w_m_f1, w_c1_f1, w_auc = 0.2, 0.4, 0.4
# n_trials=20
# epochs = 500
n_trials=50
epochs=200

# ==============================================================================
# Genetic Algorithm
# https://pygad.readthedocs.io/en/latest/
# ==============================================================================

# 假設 X_val_meta 是你所有 GNN 預測結果組成的 DataFrame
def fitness_func(ga_instance, solution, solution_idx, 
                 X_val_raw, val_ts, X_val_meta, y_val, split_threshold):
    # solution 是 GA 產生的 [1, 0, 1...] 陣列
    # selected_cols = [i for i, bit in enumerate(solution) if bit == 1]
    # print('selected_cols 1:', selected_cols)

    selected_cols = np.where(solution == 1)[0]
    # print('selected_cols2:', selected_cols2)

    if len(selected_cols) == 0: return 0
    
    # 只選取部分 GNN 的預測作為特徵
    ga_train_mask = (val_ts < split_threshold)
    ga_val_mask = (val_ts >= split_threshold)

    X_ga_train_raw = X_val_raw[ga_train_mask]
    X_ga_val_raw = X_val_raw[ga_val_mask]
    y_ga_train = y_val[ga_train_mask]
    y_ga_val = y_val[ga_val_mask]

    X_ga_train_meta = X_val_meta[ga_train_mask][:, selected_cols]
    X_ga_val_meta = X_val_meta[ga_val_mask][:, selected_cols]

    # 合併 val < 39 的部分和 meta < 39 + selected_cols 的部分 
    X_train_final = np.hstack([X_ga_train_raw, X_ga_train_meta])
    # 合併 val >= 39 的部分和 meta >= 39 + selected_cols 的部分 
    X_val_final = np.hstack([X_ga_val_raw, X_ga_val_meta])

    # 訓練一個簡單的 CatBoost 作為評估 (為了速度，可以減少 iterations)
    clf = CatBoostClassifier(iterations=20, silent=True)
    clf.fit(X_train_final, y_ga_train)
    
    # 拿 Macro F1 score
    preds = clf.predict(X_val_final)
    probs = clf.predict_proba(X_val_final)[:, 1]

    c1_f1 = f1_score(y_ga_val, preds)
    macro_f1 = f1_score(y_ga_val, preds, average='macro')
    auc = roc_auc_score(y_ga_val, probs)
    # w_m_f1, w_c1_f1, w_auc = 0.2, 0.4, 0.4
    weight_score = (w_c1_f1 * c1_f1) + (w_m_f1 * macro_f1) + (w_auc * auc)
    return weight_score


# ==============================================================================
# TPE - Optuna
# ==============================================================================

def catboost_objective(trial, data, gnns_val_probs):
    """Optuna objective function for CatBoost model optimization"""
    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate
    X_val_raw_meta = np.hstack([data.x[data.val_mask].numpy(), X_val_meta])
    y_val = data.y[data.val_mask].numpy()
    print(f"X_val_meta shape: {X_val_meta.shape}")
    print(f"X_val_final shape: {X_val_raw_meta.shape}")
    X_tr, X_ev, y_tr, y_ev = train_test_split(X_val_raw_meta, y_val, test_size=0.25, random_state=RANDOM_SEED)

    # Define search space
    # iterations = trial.suggest_int('iterations', 100, 500, step=100)
    iterations = epochs
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
    depth = trial.suggest_categorical('depth', [4, 6, 8, 10])
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
    """
    Best hyperparameters - CatBoost: FrozenTrial(number=9, state=<TrialState.COMPLETE: 1>, 
    values=[0.8925619834710744], datetime_start=datetime.datetime(2026, 3, 25, 4, 4, 49, 961337), 
    datetime_complete=datetime.datetime(2026, 3, 25, 4, 5, 18, 724769), 
    params={'learning_rate': 0.02091321857969949, 'depth': 8, 'l2_leaf_reg': 8.552105615805932}, 
    user_attrs={}, system_attrs={}, intermediate_values={}, 
    distributions={'learning_rate': FloatDistribution(high=0.3, log=True, low=0.001, step=None), 
    'depth': CategoricalDistribution(choices=(4, 6, 8, 10)), 
    'l2_leaf_reg': FloatDistribution(high=10.0, log=False, low=1.0, step=None)}, trial_id=9, value=None)
    """
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
    model_val_performance = eval_blending_catboost(y_ev, test_preds, test_probs)

    auc = model_val_performance['auc'] 
    prec = model_val_performance['class 1 precision']
    recall = model_val_performance['class 1 recall']
    f1 = model_val_performance['class 1 f1-score']

    # 設定底線
    if auc < 0.6 or prec < 0.5 or recall < 0.5:
        print("auc < 0.6 or prec < 0.5 or recall < 0.5")
        return 0.0 + (auc * 0.01)  # 極低分，但保留一點點 AUC 作為引導梯度

    # 核心：如果你很看重 Precision，就給它權重
    # 這樣就算 F1 一樣，Precision 高的也會勝出
    refined_score = f1

    # 額外懲罰區（針對你提到的 0.3 門檻）
    if prec < 0.8:
        print("prec < 0.8")
        refined_score *= 0.5  # 精確度不到 0.3，分數直接砍半
    if recall < 0.8:
        print("recall < 0.8")
        refined_score *= 0.5  # 召回率不到 0.3，分數再砍半

    return refined_score
    # return model_test_performance['macro f1-score'], model_test_performance['class 1 f1-score'], model_test_performance['auc']

# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================

# Isolation Forest Baseline
def isolation_forest_baseline(data):
    ## isolation forest baseline skip validation set
    X_train = data.x[data.train_mask | data.val_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    # Train Isolation Forest
    clf = IsolationForest(random_state=24027277, contamination=0.05)
    clf.fit(X_train)

    # Predict: 1=normal, -1=anomaly
    y_pred = clf.predict(X_test)
    y_pred = (y_pred == -1).astype(int) # Convert: 1=normal, -1=anomaly -> 1=anomaly, 0=normal
    anomaly_scores = clf.decision_function(X_test)

    test_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    test_auc = roc_auc_score(y_test, -anomaly_scores)

    # Evaluate
    # baseline_results = {
    #     'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    #     'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
    #     'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
    #     'macro_auc': roc_auc_score(y_test, -anomaly_scores),
    #     'gmean': np.sqrt(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 
    #                     (1 - precision_score(y_test, y_pred, pos_label=0, zero_division=0))),
    #     'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    #     'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
    #     'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
    #     'auc': roc_auc_score(y_test, -anomaly_scores),
    #     'accuracy': accuracy_score(y_test, y_pred),
    # }
    # print(f"Isolation Forest baseline results:")
    # for key, value in baseline_results.items():
    #     print(f"{key}: {value}")
    # print("Baseline evaluation completed")

    baseline_results = {
        'model': "Isolation Forest",
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

    return baseline_results

# GNN Model
def get_gnn(model_name, data, best_params):
    device = data.x.device
    best_params['in_channels'] = data.x.size(1)
    best_params['out_channels'] = 2
    # Initialize GNN model
    model_func = getattr(gnn_zoo, model_name, None)
    model = model_func(best_params).to(device)
    return model


def evaluate_model(model, data, mask, threshold, device):
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        out = model(x, edge_index)
        
        probs = torch.sigmoid(out[mask])
        if probs.dim() > 1 and probs.size(1) == 2:
            probs = probs[:, 1]
            
        y_pred = (probs > threshold).cpu().numpy().flatten()
        
        y_true = data.y[mask].cpu().numpy().flatten()

    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

def train_gnn(model, data, lr=0.002, alpha=0.875, weight_decay=5e-4, threshold=0.5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha, reduction='mean')

    best_val_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict()) # 儲存初始權重
    patience = 30 # 最終訓練可以給多一點點耐性
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_f1 = evaluate_model(model, data, data.val_mask, threshold, data.x.device)
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
                break

    model.load_state_dict(best_model_wts)
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
        'model': model_name.rsplit('_', 1)[0],
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
    test_report = classification_report(y_test, test_preds, zero_division=0, output_dict=True)
    test_auc = roc_auc_score(y_test, test_probs)
    # print(f"Test AUC: {auc:.4f}")
    model_test_performance = {
        'model': model_name.rsplit('_', 1)[0],
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

    
def train_and_test_gnn(model_name, data, device, best_params=None):
    # Extract best parameters or use defaults
    # in_channels = best_params.get('in_channels', None)
    # hidden_channels = best_params.get('hidden_channels', 64)
    # out_channels = best_params.get('out_channels', 2)
    # num_layers = best_params.get('num_layers', 2)
    # dropout = best_params.get('dropout', 0.5)
    # heads = best_params.get('heads', 8)
    lr = best_params.get('lr', 0.01)
    focalloss_alpha = best_params.get('focalloss_alpha', 0.875)
    # focalloss_gamma = best_params.get('focalloss_gamma', 2.0)
    weight_decay = best_params.get('weight_decay', 5e-4)
    threshold = best_params.get('threshold', 0.5)
    model = get_gnn(model_name, data, best_params)
    model.to(device)
    data_device = data.clone().to(device)
    # model = train_gnn_minibatch(model, gnn_train_loader, device, epochs, lr, focalloss_alpha, focalloss_gamma, weight_decay)
    model = train_gnn(model, data_device, lr, focalloss_alpha, weight_decay)
    return test_gnn(model, data_device, threshold)
     

# ==============================================================================
# Catboost Model - Blending
# ==============================================================================

def eval_blending_catboost(y_test, test_preds, test_probs):
     ## preformance metrics
    test_report = classification_report(y_test, test_preds, zero_division=0, output_dict=True)
    test_auc = roc_auc_score(y_test, test_probs)
    model_test_performance = {
        'model': 'CatBoost_Blending',
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
    return model_test_performance


def blending_catboost(data, gnns_val_probs, gnns_test_probs, best_params=None):
    ## process meta data
    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate
    X_test_meta = np.hstack(gnns_test_probs)
    print(f"X_val_meta shape: {X_val_meta.shape}, X_test_meta shape: {X_test_meta.shape}")

    X_val_raw_meta = np.hstack([data.x[data.val_mask].numpy(), X_val_meta])
    X_test_raw_meta = np.hstack([data.x[data.test_mask].numpy(), X_test_meta])
    print(f"X_val_final shape: {X_val_raw_meta.shape}, X_test_final shape: {X_test_raw_meta.shape}")

    # Extract best parameters or use defaults
    iterations = epochs
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
    y_val = data.y[data.val_mask].numpy()
    cat.fit(X_val_raw_meta, y_val)

    test_preds = cat.predict(X_test_raw_meta)
    test_probs = cat.predict_proba(X_test_raw_meta)[:, 1]

    return test_preds, test_probs


# ==============================================================================
# Main - Execute complete GNN anomaly detection  
# ==============================================================================
def main(dataset_name):
    print("=" * 60)
    print("Blockchain Anomaly Detection GNN Framework")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================================================
    # 1. Load Elliptic dataset
    # ========================================================================
    
    if dataset_name == 'e1':
        data = dataset_zoo.load_elliptic_data()
        
    elif dataset_name == 'e2':
        data = dataset_zoo.load_ethereum_data()
        
    # data = data.to(device)

    # ========================================================================
    # 3. Train Isolation Forest baseline - done - 暫時comment for train GNN model
    # ========================================================================
     ## 暫時comment #######################################################
    print("\n3. Training Isolation Forest baseline...")
    baseline_results = isolation_forest_baseline(data)
    df_baseline_results = pd.DataFrame([baseline_results])
    
    # ========================================================================
    # 4. Train GNN
    # 4.1. TPE 優化參數 
    # 4.2. train GNN
    # ========================================================================
    # 4. Train GCN model
    # gnn_models_list = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'APPNP', 'ChebNet', 'GCNII',
    #     'MixHop_GCN', 'MixHop_GAT', 'MixHop_GraphSAGE', 'MixHop_GIN',
    #     'MixHop_GCN_K3', 'MixHop_GAT_K3', 'MixHop_GraphSAGE_K3', 'MixHop_GIN_K3',
    #     'MixHop_GCN_K4', 'MixHop_GAT_K4', 'MixHop_GraphSAGE_K4', 'MixHop_GIN_K4',
    # ]
    # ========================================================================
    print("\n4.2. Training GNNs baseline...")
    y_val = data.y[data.val_mask].numpy()
    y_test = data.y[data.test_mask].numpy()
    gnns_val_probs, gnns_test_probs = [], []
    models_val_performance, models_test_performance = [], []

    ## get GNN ZOO all model
    gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)
    # gnn_models_list.extend([('GCN', 0), ('GAT', 0), ('GraphSAGE', 0), ('GIN', 0)])
    print('gnn models length:', len(gnn_models_list))
    for model_name, _ in gnn_models_list:

        print('\nload TPE results')
        with open(f"best_model_params/best_{dataset_name}/{model_name}_params_{dataset_name}.json", "r") as f:
            tpe_results = json.load(f)
        best_params = tpe_results["best_params"]
        print('best_params:', best_params)

        print(f"--- Training {model_name} ---")
        # best_params = tpe_results.get(model_name, {}).get("best_params", {})
        gnn_val_probs, gnn_val_preds, gnn_test_probs, gnn_test_preds = train_and_test_gnn(model_name, data, device, best_params)
        gnns_val_probs.append(gnn_val_probs.reshape(-1, 1))
        gnns_test_probs.append(gnn_test_probs.reshape(-1, 1))
        model_val_performance, model_test_performance = eval_gnn(model_name, y_val, y_test, gnn_val_probs, gnn_val_preds, gnn_test_probs, gnn_test_preds)
        models_val_performance.append(model_val_performance)
        models_test_performance.append(model_test_performance)

        print("--- done ---")

    # 1. 設置顯示所有 Column (唔好用 ... 隱藏)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    # 4. 限制小數點位數，令表格更對齊
    # pd.set_option('display.precision', 4)
    df_val_results = pd.DataFrame(models_val_performance)
    df_test_results = pd.DataFrame(models_test_performance)
    print("\nValidation Performance\n", df_val_results, "\n", "="*60, "\nTesting Performance\n", df_test_results)

    # ========================================================================
    # 5. training CatBoost
    # 5.1. GA 揀 model
    # 5.2  TPE - Optuna 優化catboost參數
    # 5.3. Blending CatBoost
    # ========================================================================
    # 5.1. GA
    # ========================================================================
    # 初始化 GA (11 個模型，所以基因長度是 11)
    print("\n5.1. GA Crossover model...")
    # 使用 validation set 同 GNN 預測結果作為 GA 的 input，因為 test set 係唔可以用嚟做 model selection
    X_val_raw = data.x[data.val_mask].numpy()
    # 需要 validation set 的 timesteps 來做 GA 的時間切分，避免 data leakage
    val_ts = None
    if dataset_name == 'e1':
        val_ts = data.timesteps[data.val_mask].numpy()
        split_threshold = 39
    else:
        val_ts = np.arange(X_val_raw.shape[0])
        split_threshold = int(len(val_ts) * 0.7)

    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate, GNN models gnn_val_probs
    bound_fitness = lambda ga_instance, solution, solution_idx: fitness_func(
        ga_instance, solution, solution_idx, 
        X_val_raw, val_ts, X_val_meta, y_val, split_threshold
    )
    # partial(fitness_func, X_data=X_val_raw_meta, y_true=y_val)
    print('gnn models length:', len(gnn_models_list))
    ga_instance = pygad.GA(
        num_generations=50, 
        num_parents_mating=5, 
        fitness_func=bound_fitness,
        sol_per_pop=20, 
        num_genes=len(gnn_models_list),
        mutation_num_genes=1,
        gene_space=[0, 1]
    ) # 二進位搜尋
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
    study.optimize(lambda trial: catboost_objective(trial, data, filtered_val_probs), n_trials=n_trials)
    # best_trials = max(study.best_trials, key=lambda t: w_m_f1 * t.values[0] + w_c1_f1 * t.values[1] + w_auc * t.values[2])
    # tpe_results[model_name] = {
    #     "macro_f1": best_trials.values[0],
    #     "c1_f1": best_trials.values[1],
    #     "auc": best_trials.values[2],
    #     "best_params": best_trials.params,
    # }
    # print("best param", model_name, "="*60, "\n", tpe_results[model_name])
    best_trial = study.best_trial
    cat_best_params = best_trial.params
    
    # cat_best_params = study.best_params
    # cat_best_value = study.best_value
    print(f"Best hyperparameters - CatBoost: {cat_best_params}")
    # print(f"Best score - auc?: {cat_best_value}")
    # for trial in study.best_trials:
    #     print(f"Trial #{trial.number} Values: {trial.values}")
    # study = optuna.create_study(direction='maximize')
    # study.optimize(lambda trial: catboost_objective(trial, data, gnns_val_probs, gnns_test_probs), n_trials=n_trials)
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
    cat_test_preds, cat_test_probs = blending_catboost(data, filtered_val_probs, filtered_test_probs, cat_best_params)
    # cat_test_preds, cat_test_probs = blending_catboost(data, gnns_val_probs, gnns_test_probs)
    cat_test_performance = eval_blending_catboost(y_test, cat_test_preds, cat_test_probs)
    models_test_performance.append(cat_test_performance)
    df_cat_preformance = pd.DataFrame([cat_test_performance])
    final_df_test_results = pd.concat([df_baseline_results, df_test_results, df_cat_preformance], ignore_index=True)
    print("\n", "="*60, "\n", final_df_test_results)


    # ========================================================================
    # 6. Chart
    # ========================================================================
    plot_model_comparison(final_df_test_results)
    #  

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: elliptic or ethereum')
    args = parser.parse_args()
    main(args.dataset)