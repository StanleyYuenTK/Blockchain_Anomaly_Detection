import gc
import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import json
import optuna

# GNN Models
import inspect
import gnn_zoo
import dataset_zoo

## Focal Loss https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
from kornia.losses import FocalLoss
import argparse

RANDOM_SEED = 24027277
n_trials=100
epochs = 500
device = 'cuda'
model_name = 'MixHop_Model'
# ==============================================================================
# TPE - Optuna
# ==============================================================================
def objective(trial, data):
    # model
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.45, 0.65)
    powers = trial.suggest_categorical("powers", ["(0, 1, 2)", "(0, 1, 2, 3)"])
    # powers = eval(powers_option)

    # training 
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.65, 0.85) 
    gamma = trial.suggest_float("gamma", 2.0, 3.5)

    gnn_best_params = {
        'in_channels': data.x.size(1),
        'out_channels': 2,
        'hidden_channels':hidden_channels,
        'dropout':dropout,
        'powers':powers
    }

    model = get_gnn(model_name, gnn_best_params, device) 
    data = data.to(device)
    
    # model = build_mixhop_model(data.num_features, hidden_channels, powers, dropout).to(data.x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

    # 3. 訓練迴圈
    model.train()
    for epoch in range(epochs):  # 搜索時可縮短 epoch
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # 中間評估（用於剪枝）
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index)[data.val_mask].argmax(dim=1)
                val_f1 = f1_score(data.y[data.val_mask].cpu(), pred.cpu(), average='macro')
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            model.train()

    # 4. 最終評估
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        logits = out.argmax(dim=1)
        train_f1 = f1_score(data.y[data.train_mask].cpu(), logits[data.train_mask].cpu(), average='macro')
        val_f1 = f1_score(data.y[data.val_mask].cpu(), logits[data.val_mask].cpu(), average='macro')

    gap = abs(train_f1 - val_f1)
    final_score = val_f1 - 0.5 * gap

    return final_score

# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================

# GNN Model
def get_gnn(model_name, best_params, device):
    model_func = getattr(gnn_zoo, model_name, None)
    model = model_func(best_params).to(device)
    return model

   
# ==============================================================================
# Main - Execute complete GNN anomaly detection  
# ==============================================================================
def main(dataset_name="e1"):
    print("=" * 60, "\nGNN TPE - Optuna\n", "=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================================================
    # 1. Load dataset
    # ========================================================================
    if dataset_name == 'e1':
        data = dataset_zoo.load_elliptic_data()
    elif dataset_name == 'e2':
        data = dataset_zoo.load_ethereum_data()
    
    # ========================================================================
    # 2. TPE - Optuna - done
    # ========================================================================
    print("\n2. TPE - Optuna GNN models...")

    print(model_name, "="*60)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, data), n_trials=n_trials)
    
    best_trial = study.best_trial
    tpe_result = {"c1_f1": best_trial.value,"best_params": best_trial.params,}
    print("best param", model_name, "="*60, "\n", tpe_result)

    with open(f"best_model_params/{model_name}_params_{dataset_name}.json", "w") as f:
        json.dump(tpe_result, f, indent=4)
    
    print(f"saved {model_name}_tpe_params")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: e1 or e2')
    args = parser.parse_args()
    main(args.dataset)