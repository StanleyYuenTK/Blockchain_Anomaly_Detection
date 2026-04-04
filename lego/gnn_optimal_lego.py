"""
The Hong Kong Polytechnic University
Student ID: 24027277d
Name: Yuen Tsz Ki

Use TPE to fine-tune each model and save the optimal parameters.
"""

import gc
import json
import argparse
import inspect

import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# dataset
import dataset_zoo

# GNN Models
from gnn_lego import Lego_GNN

# optimal
import optuna
from kornia.losses import FocalLoss



RANDOM_SEED = 24027277
n_trials=50
epochs=200
threshold=0.3

# Set device
device = 'cuda'
# ==============================================================================
# TPE - Optuna
# ==============================================================================
def gnn_objective(dataset_name, trial, model_name, data):
    try:

        num_comm = data.comm_id.max().item() + 1

     
        # define search space for model
        hidden_channels = trial.suggest_categorical('hidden_channels', [8, 16, 32, 64])
        dropout = trial.suggest_float('dropout', 0.4, 0.6, log=False)
        powers = trial.suggest_categorical("powers", ["(0, 1, 2, 3)", "(0, 1, 2, 4)", "(1, 2, 3, 4)", "(1, 3, 5)"])
        K = trial.suggest_int('K', 2, 6)
        alpha = trial.suggest_float('alpha', 0.2, 0.5) 
        gnn_best_params = {
            'in_channels': data.x.size(1),
            'out_channels': 2,
            'hidden_channels': hidden_channels,
            'dropout': dropout,
            'powers':powers,
            'K':K,
            'alpha':alpha
        }
        
        model = Lego_GNN(num_communities=num_comm, best_params=gnn_best_params) 
 
        # define search space for training, testing parameters
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        focalloss_alpha = trial.suggest_float('focalloss_alpha', 0.5, 0.9, step=0.1)

        threshold=0.5


        # training and testing and evaluate --------------
        model = train_gnn_fullbatch(trial, data, threshold, model, 
            epochs=epochs, lr=lr, alpha=focalloss_alpha, weight_decay=weight_decay
        )

        val_probs, val_preds, test_probs, test_preds, val_y, test_y = test_gnn(
            model, data, threshold=threshold
        )
    
        model_val_performance, model_test_performance = eval_gnn(
            model_name, val_y, test_y, val_probs, val_preds, test_probs, test_preds
        )
        ## macro f1 score, class 1 f1-score, auc
                
        auc = model_val_performance['auc'] 
        prec = model_val_performance['class 1 precision']
        recall = model_val_performance['class 1 recall']
        f1 = model_val_performance['class 1 f1-score']
        trial.set_user_attr("precision", prec)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("test_f1", f1)
    
        if auc < 0.6 or prec < 0.4 or recall < 0.4:
            return f1 *0.1

        refined_score = 0.7 * f1 + 0.2 * prec + 0.1 * recall

        if prec < 0.5:
            refined_score *= 0.5
        if recall < 0.5:
            refined_score *= 0.5

        return refined_score
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================
def evaluate_model(model, data, mask, threshold):
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        out = model(data)
        
        probs = torch.sigmoid(out[mask])
        if probs.dim() > 1 and probs.size(1) == 2:
            probs = probs[:, 1]
            
        y_pred = (probs > threshold).cpu().numpy().flatten()
        y_true = data.y[mask].cpu().numpy().flatten()

    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_gnn_fullbatch(trial, data, threshold, model, epochs, lr, alpha, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha, reduction='mean')
    model.to(device)

    best_val_f1 = 0
    patience = 100
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        val_f1 = evaluate_model(model, data, data.val_mask, threshold)
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            counter = 0 
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
    return model


def test_gnn(model, data, threshold=0.25): 
    model.eval()
    model.to(device)

    with torch.no_grad():
        out = model(data)
        probs = torch.softmax(out, dim=1)[:, 1]
        
        val_probs = probs[data.val_mask].cpu().numpy()
        test_probs = probs[data.test_mask].cpu().numpy()
        val_y = data.y[data.val_mask].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()

    val_preds = (val_probs > threshold).astype(int)
    test_preds = (test_probs > threshold).astype(int)

    return val_probs, val_preds, test_probs, test_preds, val_y, test_y

def eval_gnn(model_name, y_val, y_test, val_probs, val_preds, test_probs, test_preds):
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


# ==============================================================================
# Main - Execute complete GNN anomaly detection  
# ==============================================================================
def main(dataset_name="e1"):
    print("=" * 60, "\nGNN TPE - Optuna\n", "=" * 60)

    # ========================================================================
    # 1. Load dataset
    # ========================================================================
    if dataset_name == 'e1':
        data = dataset_zoo.load_elliptic_data()
    elif dataset_name == 'e2':
        data = dataset_zoo.load_ethereum_data()
    data = data.to(device)

    # ========================================================================
    # TPE - Optuna - optimize GNN model
    # ========================================================================
    print("\nTPE - Optuna GNN models...")
    # gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)

    # for model_name, _ in gnn_models_list:
        # print(model_name, "="*60)
    model_name = "lego"
    study = optuna.create_study(direction='maximize', load_if_exists=False, pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(lambda trial: gnn_objective(dataset_name, trial, model_name, data), n_trials=n_trials)
    
    best_trial = study.best_trial
    tpe_result = {"c1_f1": best_trial.value,"best_params": best_trial.params,}
    print("best param", model_name, "="*60, "\n", tpe_result)

    with open(f"best_model_params/{model_name}_params_{dataset_name}.json", "w") as f:
        json.dump(tpe_result, f, indent=4)
    
    print(f"saved {model_name}_tpe_params")

    df = study.trials_dataframe()
    print(df[['value', 'user_attrs_precision', 'user_attrs_recall', 'user_attrs_auc', 'user_attrs_test_f1']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: e1 or e2')
    args = parser.parse_args()
    main(args.dataset)