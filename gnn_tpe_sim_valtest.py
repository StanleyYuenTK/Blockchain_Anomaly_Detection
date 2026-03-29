
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, confusion_matrix

# model
from sklearn.ensemble import IsolationForest

from catboost import CatBoostClassifier

# Optimization
import optuna
import pygad

import json
import copy

# visualiz
from visualization_tools import plot_model_comparison

# GNN Models
import inspect
import gnn_zoo
import dataset_zoo

## Focal Loss https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
from kornia.losses import FocalLoss
import argparse


RANDOM_SEED = 24027277
w_m_f1, w_c1_f1, w_auc = 0.4, 0.5, 0.1
n_trials=50
epochs=200
MIN_GA_MODELS = 1


# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================

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
    best_model_wts = copy.deepcopy(model.state_dict()) 
    patience = 100
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
    val_report = classification_report(y_val, val_preds, zero_division=0, output_dict=True)
    val_auc = roc_auc_score(y_val, val_probs)
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

    
def train_and_test_gnn(model_name, data, device, best_params={}):
    lr = best_params.get('lr', 0.01)
    focalloss_alpha = best_params.get('focalloss_alpha', 0.875)
    weight_decay = best_params.get('weight_decay', 5e-4)
    threshold = best_params.get('threshold', 0.5)

    model = get_gnn(model_name, data, best_params)
    model.to(device)
    data_device = data.clone().to(device)
    model = train_gnn(model, data_device, lr, focalloss_alpha, weight_decay)
    return test_gnn(model, data_device, threshold)
     

# ==============================================================================
# Main - Execute complete GNN anomaly detection  
# ==============================================================================
def main(dataset_name):
    print("=" * 60, "\nGNN TPE Simulate Validataion & tseting\n", "=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    # ========================================================================
    # 1. Load Elliptic dataset
    # ========================================================================
    ver = ''
    if 'nol' in dataset_name:
        dataset_name = "e1" if 'e1' in dataset_name else 'e2' 
        ver = 'nol'

    if dataset_name == 'e1':
        print("dataset:", dataset_name, 'ver', ver)
        data = dataset_zoo.load_elliptic_data(ver=ver)    
        
    elif dataset_name == 'e2':
        print("dataset:", dataset_name, 'ver', ver)
        data = dataset_zoo.load_ethereum_data(ver=ver)

    # ========================================================================
    # Train GCN model
    # ========================================================================
    y_val = data.y[data.val_mask].numpy()
    y_test = data.y[data.test_mask].numpy()
    gnns_val_probs, gnns_test_probs = [], []
    models_val_performance, models_test_performance = [], []

    ## get GNN ZOO all model
    gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)
    for model_name, _ in gnn_models_list:
        file_path = f"best_model_params/{model_name}_params_{dataset_name}.json"
        print("\n", "-"*60, f"Training {model_name}", "-"*60, "\nfile Path: ", file_path)

        with open(file_path, "r") as f:
            tpe_results = json.load(f)
        best_params = tpe_results["best_params"]
        print('best_params:', best_params)

        gnn_val_probs, gnn_val_preds, gnn_test_probs, gnn_test_preds = train_and_test_gnn(model_name, data, device, best_params)
        gnns_val_probs.append(gnn_val_probs.reshape(-1, 1))
        gnns_test_probs.append(gnn_test_probs.reshape(-1, 1))
        model_val_performance, model_test_performance = eval_gnn(model_name, y_val, y_test, gnn_val_probs, gnn_val_preds, gnn_test_probs, gnn_test_preds)
        models_val_performance.append(model_val_performance)
        models_test_performance.append(model_test_performance)

        print("--- done ---")

    df_val_results = pd.DataFrame(models_val_performance)
    df_test_results = pd.DataFrame(models_test_performance)
    print("\nValidation Performance\n", df_val_results, "\n", "="*60, "\nTesting Performance\n", df_test_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: elliptic or ethereum')
    args = parser.parse_args()
    main(args.dataset)