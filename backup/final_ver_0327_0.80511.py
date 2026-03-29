
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
# Genetic Algorithm
# https://pygad.readthedocs.io/en/latest/
# ==============================================================================
def fitness_func(ga_instance, solution, solution_idx, 
                 X_val_raw, val_ts, X_val_meta, y_val, split_threshold):

    selected_cols = np.where(solution == 1)[0]

    # if len(selected_cols) == 0: return 0
    if len(selected_cols) < MIN_GA_MODELS:
        return 0
    
    ga_train_mask = (val_ts < split_threshold)
    ga_val_mask = (val_ts >= split_threshold)

    X_ga_train_raw = X_val_raw[ga_train_mask]
    X_ga_val_raw = X_val_raw[ga_val_mask]
    y_ga_train = y_val[ga_train_mask]
    y_ga_val = y_val[ga_val_mask]

    X_ga_train_meta = X_val_meta[ga_train_mask][:, selected_cols]
    X_ga_val_meta = X_val_meta[ga_val_mask][:, selected_cols]

    X_train_final = np.hstack([X_ga_train_raw, X_ga_train_meta])
    X_val_final = np.hstack([X_ga_val_raw, X_ga_val_meta])

    cat = CatBoostClassifier(
        iterations=epochs,
        loss_function='Logloss',
        eval_metric='Logloss',
        early_stopping_rounds=(epochs*0.2),
        random_seed=RANDOM_SEED,
        auto_class_weights='Balanced',

        # randome select 80% data
        bootstrap_type='Bernoulli',
        subsample=0.8,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10,
        silent=True,
        allow_writing_files=False,
    )
    # cat = CatBoostClassifier(
    #     iterations=20,
    #     learning_rate=0.1,
    #     depth=5,
    #     bootstrap_type='Bernoulli',
    #     subsample=0.8,
    #     # task_type='GPU'
    #     # devices='0',
    #     silent=True,
    #     allow_writing_files=False
    # )
    cat.fit(X_train_final, y_ga_train, eval_set=(X_val_final, y_ga_val))
    
    preds = cat.predict(X_val_final)
    probs = cat.predict_proba(X_val_final)[:, 1]

    c1_f1 = f1_score(y_ga_val, preds)
    macro_f1 = f1_score(y_ga_val, preds, average='macro')
    auc = roc_auc_score(y_ga_val, probs)
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
    # iterations = 500
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.5, log=True)
    depth = trial.suggest_int('depth', 4, 10)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True)
    # cat = CatBoostClassifier(
    #     iterations,
    #     learning_rate,
    #     depth,
    #     l2_leaf_reg,
    #     loss_function='Logloss',
    #     eval_metric='F1',
    #     task_type='GPU',
    #     devices='0',
    #     early_stopping_rounds=(iterations*0.2),
    #     random_seed=RANDOM_SEED,
    #     auto_class_weights='Balanced',
    #     verbose=100,
    # )
    cat = CatBoostClassifier(
        iterations=epochs,
        loss_function='Logloss',
        eval_metric='Logloss',
        early_stopping_rounds=(epochs*0.2),
        random_seed=RANDOM_SEED,
        auto_class_weights='Balanced',

        # randome select 80% data
        bootstrap_type='Bernoulli',
        subsample=0.8,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        silent=True,
        allow_writing_files=False,
    )


    # cat.fit(X_tr, y_tr)
    cat.fit(X_tr, y_tr, eval_set=(X_ev, y_ev), use_best_model=True)

    test_preds = cat.predict(X_ev)
    test_probs = cat.predict_proba(X_ev)[:, 1]
    model_val_performance = eval_stacking_catboost(y_ev, test_preds, test_probs)

    auc = model_val_performance['auc'] 
    prec = model_val_performance['class 1 precision']
    recall = model_val_performance['class 1 recall']
    f1 = model_val_performance['class 1 f1-score']

    if auc < 0.6 or prec < 0.5 or recall < 0.5:
        print("auc < 0.6 or prec < 0.5 or recall < 0.5")
        return 0.0 + (auc * 0.01)

    refined_score = f1

    if prec < 0.8:
        print("prec < 0.8")
        refined_score *= 0.5  
    if recall < 0.8:
        print("recall < 0.8")
        refined_score *= 0.5 

    return refined_score


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
    clf = IsolationForest(random_state=RANDOM_SEED, contamination=0.05)
    clf.fit(X_train)

    # Predict: 1=normal, -1=anomaly
    y_pred = clf.predict(X_test)
    y_pred = (y_pred == -1).astype(int) # Convert: 1=normal, -1=anomaly -> 1=anomaly, 0=normal
    anomaly_scores = clf.decision_function(X_test)

    test_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    test_auc = roc_auc_score(y_test, -anomaly_scores)

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

    
def train_and_test_gnn(model_name, data, device, best_params=None):
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
# Catboost Model - stacking
# ==============================================================================

def eval_stacking_catboost(y_test, test_preds, test_probs):
     ## preformance metrics
    test_report = classification_report(y_test, test_preds, zero_division=0, output_dict=True)
    test_auc = roc_auc_score(y_test, test_probs)
    model_test_performance = {
        'model': 'CatBoost_Stacking',
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


def stacking_catboost(data, gnns_val_probs, gnns_test_probs, best_params={}):
    ## process meta data
    X_val_meta = np.hstack(gnns_val_probs)   # hstack same as concatenate
    X_test_meta = np.hstack(gnns_test_probs)
    print(f"X_val_meta shape: {X_val_meta.shape}, X_test_meta shape: {X_test_meta.shape}")

    X_val_raw_meta = np.hstack([data.x[data.val_mask].numpy(), X_val_meta])
    X_test_raw_meta = np.hstack([data.x[data.test_mask].numpy(), X_test_meta])
    print(f"X_val_final shape: {X_val_raw_meta.shape}, X_test_final shape: {X_test_raw_meta.shape}")

    # Extract best parameters or use defaults
    iterations = epochs
    learning_rate = best_params.get('learning_rate', 0.03)
    depth = best_params.get('depth', 4)
    l2_leaf_reg = best_params.get('l2_leaf_reg', 10.0)

    ## create catboost model and train and predict
    cat = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function='Logloss',
        eval_metric='F1',
        task_type='GPU',
        devices='0',
        # early_stopping_rounds=30,
        random_seed=RANDOM_SEED,
        auto_class_weights='Balanced',
        verbose=100,
    )
    print(cat.get_params())

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
    print("=" * 60, "\nBlockchain Anomaly Detection GNN Framework\n", "=" * 60)


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
    # Train Isolation Forest baseline
    # ========================================================================
    print("\nTraining Isolation Forest baseline...")
    baseline_results = isolation_forest_baseline(data)
    df_baseline_results = pd.DataFrame([baseline_results])
    print("--- done ---")

    # ========================================================================
    # Train GNN
    # TPE
    # train GNN
    # ========================================================================
    # Train GCN model
    # gnn_models_list = []
    # ========================================================================
    print("\nTraining GNNs baseline...")
    y_val = data.y[data.val_mask].numpy()
    y_test = data.y[data.test_mask].numpy()
    gnns_val_probs, gnns_test_probs = [], []
    models_val_performance, models_test_performance = [], []

    ## get GNN ZOO all model
    gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)
    print('\ngnn models length:', len(gnn_models_list))
    for model_name, _ in gnn_models_list:
        print(f"\n--- Training {model_name} ---")

        with open(f"best_model_params/best_{dataset_name}/{model_name}_params_{dataset_name}.json", "r") as f:
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


    # ========================================================================
    # training CatBoost
    # GA select model
    # TPE - Optuna optimal catboost hyperparameter
    # Stacking CatBoost
    # ========================================================================
    # GA
    # ========================================================================
    print("\nGA Crossover model...")
    X_val_raw = data.x[data.val_mask].numpy()
    val_ts = None
    if 'e1' in dataset_name:
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
    print('gnn models length:', len(gnn_models_list))
    ga_instance = pygad.GA(
        num_generations=10, 
        sol_per_pop=10, 
        num_parents_mating=5, 
        fitness_func=bound_fitness,
        num_genes=len(gnn_models_list),
        parent_selection_type='tournament',
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes=25,
        mutation_num_genes=1,
        gene_space=[0, 1]
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")

    print(f"\nTPE - Optuna - CatBoost with \nGA selection: {solution}...")
    selected_indices = [i for i, bit in enumerate(solution) if bit == 1]
    filtered_val_probs = [gnns_val_probs[i] for i in selected_indices]
    filtered_test_probs = [gnns_test_probs[i] for i in selected_indices]
    print(f"Using {len(selected_indices)} models selected by GA for final Stacking.")
    

    # ========================================================================
    # TPE - Optuna - CatBoost
    # ========================================================================
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: catboost_objective(trial, data, filtered_val_probs), n_trials=n_trials)
    best_trial = study.best_trial
    cat_best_params = best_trial.params
    print(f"Best hyperparameters - CatBoost: {cat_best_params}")
   
    # ========================================================================
    # stacking ensemble model
    # ========================================================================
    print("\nStacking CatBoost...")
    cat_test_preds, cat_test_probs = stacking_catboost(data, filtered_val_probs, filtered_test_probs, cat_best_params)
    cat_test_performance = eval_stacking_catboost(y_test, cat_test_preds, cat_test_probs)
    models_test_performance.append(cat_test_performance)
    df_cat_preformance = pd.DataFrame([cat_test_performance])
    final_df_test_results = pd.concat([df_baseline_results, df_test_results, df_cat_preformance], ignore_index=True)
    print("\n", "="*60, "\n", final_df_test_results)

    # ========================================================================
    # Chart
    # ========================================================================
    plot_model_comparison(final_df_test_results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: elliptic or ethereum')
    args = parser.parse_args()
    main(args.dataset)