import gc
import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
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
n_trials=30
batch_size=4096 # 2048, 8192, 16384, 32768, 65536


# ==============================================================================
# NeighborLoader for mini-batch training
# =============================================================================
def get_train_loader(data, batch_size=512, num_neighbors=[25, 10]):
    sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

# ==============================================================================
# TPE - Optuna
# ==============================================================================

def gnn_objective(dataset_name, trial, model_name, data, batch_size, device):
    """Optuna objective function for GNN model optimization"""
    try:
        ## for create model ==============================================================================
        gnn_best_params = {
            'in_channels': data.x.size(1),
            'out_channels': 2,
        }
        if dataset_name == "e1":
            if model_name in ["APPNP_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 48])
                dropout = trial.suggest_float('dropout', 0.2, 0.5)            
                K = trial.suggest_int('K', 5, 20, step=5) # 傳播步數
                alpha = trial.suggest_float('alpha', 0.1, 0.2) # 傳送概率
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'K': K,
                    'alpha': alpha
                }

            elif model_name in ["ChebNet_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [64, 80, 96, 112])
                dropout = trial.suggest_float('dropout', 0.30, 0.45, log=False)          
                K = trial.suggest_int('K', 2, 3) 
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'K': K,
                }

            elif model_name in ["GAT_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [256, 384, 512])
                dropout = trial.suggest_float('dropout', 0.25, 0.45, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                heads = trial.suggest_categorical('heads', [4, 8])
                jk = trial.suggest_categorical('jk', ['max', 'cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'heads':heads,
                    'jk':jk
                }

            elif model_name in ["GCN_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [128, 256, 512])
                dropout = trial.suggest_float('dropout', 0.25, 0.45, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                jk = trial.suggest_categorical('jk', ['max', 'cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'jk':jk
                }

            elif model_name in ["GIN_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [128, 256])
                dropout = trial.suggest_float('dropout', 0.45, 0.7, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                jk = trial.suggest_categorical('jk', ['max', 'cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'jk':jk
                }

            elif model_name in ["GraphSAGE_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [256, 512])
                dropout = trial.suggest_float('dropout', 0.4, 0.6, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                jk = trial.suggest_categorical('jk', ['cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'jk':jk
                }

            elif model_name =="MixHop_Model":
                hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256, 512])
                dropout = trial.suggest_float('dropout', 0.3, 0.6, log=False)
                ## power optuna not support tuple, list...
                power_map = {"(0, 1, 2)": (0, 1, 2), "(0, 1, 2, 3)": (0, 1, 2, 3), "(0, 1, 2, 3, 4)": (0, 1, 2, 3, 4)}
                power_key = trial.suggest_categorical("powers_config", ["(0, 1, 2)", "(0, 1, 2, 3)", "(0, 1, 2, 3, 4)"])
                powers = power_map[power_key]
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'powers':powers
                }
            
            gnn_best_params = gnn_best_params | params
            model = get_gnn(model_name, data, gnn_best_params, device)

            # for training and testing ==============================================================================
            
            
            
            if model_name in ["APPNP_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.65, log=False)
                num_neighbors_map = {"[15, 10]": [15, 10], "[25, 15]": [25, 15], "[30, 20]":[30, 20]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[15, 10]", "[25, 15]", "[30, 20]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]       
                
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9, 0.95])
                lr_max = 1e-3 if focalloss_alpha > 0.9 else 5e-3
                lr = trial.suggest_float('lr', 1e-4, lr_max, log=True)
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["ChebNet_Model"]:
                threshold = trial.suggest_float('threshold', 0.45, 0.55, log=False)
                num_neighbors_map = {"[50, 25]":[50, 25], "[64, 32]": [64, 32]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[50, 25]", "[64, 32]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]

                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9, 0.95])
                lr_max = 1e-3 if focalloss_alpha > 0.9 else 5e-3
                lr = trial.suggest_float('lr', 1e-4, lr_max, log=True)
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GAT_Model"]:
                threshold = trial.suggest_float('threshold', 0.4, 0.5, log=False)
                num_neighbors_map = {"[64, 64]": [64, 64], "[80, 40]": [80, 40], "[100, 50]": [100, 50]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[64, 64]", "[80, 40]", "[100, 50]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 2e-4, 8e-4, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.85, 0.9, 0.95])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.5, 5.5)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GCN_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                num_neighbors_map = {"[15, 10]": [15, 10], "[20, 10]": [20, 10], "[25, 15]": [25, 15]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[15, 10]", "[20, 10]", "[25, 15]"])

                # num_neighbors_map = {"[64, 64]": [64, 64], "[80, 40]": [80, 40], "[100, 50]": [100, 50]}
                # num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[64, 64]", "[80, 40]", "[100, 50]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 8e-4, 3e-3, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.85, 0.9, 0.95])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.5, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GIN_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                num_neighbors_map = {"[20, 10]": [20, 10], "[25, 15]": [25, 15]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[20, 10]", "[25, 15]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 1e-3, 3e-3, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.85, 0.9])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 5.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GraphSAGE_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                num_neighbors_map = {
                    "[10, 5]": [10, 5], 
                    "[15, 10]": [15, 10], 
                    "[25, 15]": [25, 15]
                }
                n_n_m_k = list(num_neighbors_map.keys())
                num_neighbors_key = trial.suggest_categorical('num_neighbors', n_n_m_k)
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.9, 0.95])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 5.0, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-5, log=True)

            elif model_name =="MixHop_Model":
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                
                num_neighbors_map = {"[15, 10]": [15, 10], "[20, 15]": [20, 15]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[15, 10]", "[20, 15]"])
                
                num_neighbors = num_neighbors_map[num_neighbors_key]

                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9])
                lr = trial.suggest_float('lr', 1e-4, 3e-4, log=True)
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 5.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        else:

            if model_name in ["APPNP_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 48])
                dropout = trial.suggest_float('dropout', 0.2, 0.5)            
                K = trial.suggest_int('K', 5, 20, step=5) # 傳播步數
                alpha = trial.suggest_float('alpha', 0.1, 0.2) # 傳送概率
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'K': K,
                    'alpha': alpha
                }

            elif model_name in ["ChebNet_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 80, 96, 112])
                dropout = trial.suggest_float('dropout', 0.30, 0.45, log=False)          
                K = trial.suggest_int('K', 2, 3) 
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'K': K,
                }

            elif model_name in ["GAT_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
                dropout = trial.suggest_float('dropout', 0.3, 0.6, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                heads = trial.suggest_categorical('heads', [4, 8])
                jk = trial.suggest_categorical('jk', ['max', 'cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'heads':heads,
                    'jk':jk
                }

            elif model_name in ["GCN_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
                dropout = trial.suggest_float('dropout', 0.25, 0.45, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                jk = trial.suggest_categorical('jk', ['max', 'cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'jk':jk
                }

            elif model_name in ["GIN_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
                dropout = trial.suggest_float('dropout', 0.45, 0.7, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                jk = trial.suggest_categorical('jk', ['max', 'cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'jk':jk
                }

            elif model_name in ["GraphSAGE_Model"]:
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
                dropout = trial.suggest_float('dropout', 0.4, 0.6, log=False)
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                jk = trial.suggest_categorical('jk', ['cat'])
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'num_layers': num_layers,
                    'jk':jk
                }

            elif model_name =="MixHop_Model":
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
                dropout = trial.suggest_float('dropout', 0.3, 0.6, log=False)
                ## power optuna not support tuple, list...
                power_map = {"(0, 1, 2)": (0, 1, 2), "(0, 1, 2, 3)": (0, 1, 2, 3), "(0, 1, 2, 3, 4)": (0, 1, 2, 3, 4)}
                power_key = trial.suggest_categorical("powers_config", ["(0, 1, 2)", "(0, 1, 2, 3)", "(0, 1, 2, 3, 4)"])
                powers = power_map[power_key]
                params = {
                    'hidden_channels': hidden_channels,
                    'dropout': dropout,
                    'powers':powers
                }
            
            gnn_best_params = gnn_best_params | params
            model = get_gnn(model_name, data, gnn_best_params, device)

            # for training and testing ==============================================================================
            
            
            
            if model_name in ["APPNP_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.65, log=False)
                num_neighbors_map = {"[15, 10]": [15, 10], "[25, 15]": [25, 15], "[30, 20]":[30, 20]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[15, 10]", "[25, 15]", "[30, 20]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]       
                
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9, 0.95])
                lr_max = 1e-3 if focalloss_alpha > 0.9 else 5e-3
                lr = trial.suggest_float('lr', 1e-4, lr_max, log=True)
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["ChebNet_Model"]:
                threshold = trial.suggest_float('threshold', 0.45, 0.55, log=False)
                num_neighbors_map = {"[50, 25]":[50, 25], "[64, 32]": [64, 32]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[50, 25]", "[64, 32]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]

                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9, 0.95])
                lr_max = 1e-3 if focalloss_alpha > 0.9 else 5e-3
                lr = trial.suggest_float('lr', 1e-4, lr_max, log=True)
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GAT_Model"]:
                threshold = trial.suggest_float('threshold', 0.4, 0.6, log=False)
                num_neighbors_map = {"[64, 64]": [64, 64], "[80, 40]": [80, 40], "[100, 50]": [100, 50]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[64, 64]", "[80, 40]", "[100, 50]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                # lr = trial.suggest_float('lr', 2e-4, 8e-4, log=True)
                lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.85, 0.9, 0.95])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.5, 5.5)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GCN_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                num_neighbors_map = {"[15, 10]": [15, 10], "[20, 10]": [20, 10], "[25, 15]": [25, 15]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[15, 10]", "[20, 10]", "[25, 15]"])

                # num_neighbors_map = {"[64, 64]": [64, 64], "[80, 40]": [80, 40], "[100, 50]": [100, 50]}
                # num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[64, 64]", "[80, 40]", "[100, 50]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 8e-4, 3e-3, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.85, 0.9, 0.95])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.5, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GIN_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                num_neighbors_map = {"[20, 10]": [20, 10], "[25, 15]": [25, 15]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[20, 10]", "[25, 15]"])
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 1e-3, 3e-3, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.85, 0.9])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 5.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            elif model_name in ["GraphSAGE_Model"]:
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                num_neighbors_map = {
                    "[10, 5]": [10, 5], 
                    "[15, 10]": [15, 10], 
                    "[25, 15]": [25, 15]
                }
                n_n_m_k = list(num_neighbors_map.keys())
                num_neighbors_key = trial.suggest_categorical('num_neighbors', n_n_m_k)
                num_neighbors = num_neighbors_map[num_neighbors_key]    

                lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.9, 0.95])
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 5.0, 6.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-5, log=True)

            elif model_name =="MixHop_Model":
                threshold = trial.suggest_float('threshold', 0.5, 0.6, log=False)
                
                num_neighbors_map = {"[15, 10]": [15, 10], "[20, 15]": [20, 15]}
                num_neighbors_key = trial.suggest_categorical('num_neighbors', ["[15, 10]", "[20, 15]"])
                
                num_neighbors = num_neighbors_map[num_neighbors_key]

                focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9])
                lr = trial.suggest_float('lr', 1e-4, 3e-4, log=True)
                focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 5.0)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # training model --------------
        # define search space for training, testing parameters
        loader = get_train_loader(data, batch_size, num_neighbors)
        
        # training and testing and evaluate
        model = train_gnn_minibatch(trial, data, threshold, model, loader, device, 
            epochs=100, lr=lr, alpha=focalloss_alpha, gamma=focalloss_gamma, weight_decay=weight_decay
        )
        val_probs, val_preds, test_probs, test_preds, val_y, test_y = test_gnn(
            model, data, batch_size, num_neighbors, device, threshold=threshold
        )
    
        model_val_performance, model_test_performance = eval_gnn(
            model_name, val_y, test_y, val_probs, val_preds, test_probs, test_preds
        )
        ## macro f1 score, class 1 f1-score, auc
        if model_val_performance['auc'] < 0.7:
            return 0.0
        return model_val_performance['class 1 f1-score']
    
    finally:
        # 確保不管成功失敗都清理顯存
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================

# GNN Model
def get_gnn(model_name, data, best_params, device):
    # device = data.x.device
    best_params['in_channels'] = data.x.size(1)
    best_params['out_channels'] = 2
    # Initialize GNN model
    model_func = getattr(gnn_zoo, model_name, None)
    model = model_func(best_params).to(device)
    return model


def evaluate_model(model, data, mask, threshold, device):
    model.eval()
    with torch.no_grad():
        # 確保 data 相關部分在正確的 device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        out = model(x, edge_index)
        
        # 只取驗證集部分的輸出
        # 假設輸出是 [N, 2]，取 index 1 代表異常類的機率
        probs = torch.sigmoid(out[mask])
        if probs.dim() > 1 and probs.size(1) == 2:
            probs = probs[:, 1]
            
        # 轉成 1D 的預測結果
        y_pred = (probs > threshold).cpu().numpy().flatten() # 加上 flatten() 確保是 1D
        
        # 轉成 1D 的真實標籤
        y_true = data.y[mask].cpu().numpy().flatten() # 加上 flatten() 確保是 1D

    # 現在兩者都是 1D numpy array，f1_score 就不會報錯了
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def train_gnn_minibatch(trial, data, threshold, model, loader, device, epochs=100, lr=0.002, alpha=0.875, gamma=2.0, weight_decay=5e-4):
    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # --- 重點：每一輪結束後，做一次驗證 ---
        val_f1 = evaluate_model(model, data, mask=data.val_mask, threshold=threshold, device=device)
        
        # --- 重點：向 Optuna 匯報成績 ---
        trial.report(val_f1, epoch)
        
        # --- 重點：檢查是否需要剪枝 (Optuna 的自動早停) ---
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return model

def test_gnn(model, data, batch_size, num_neighbors, device, threshold=0.25): 
    model.eval()
    inference_nodes = torch.where(data.val_mask | data.test_mask)[0]
    inference_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=inference_nodes,
        num_workers=8,
        shuffle=False # 保持順序以便後續拼裝
    )
    val_probs_list, test_probs_list = [], []
    val_y_list, test_y_list = [], []

    with torch.no_grad():
        for batch in inference_loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]
        
            out_seed = probs[:batch.batch_size]
            val_mask_seed = batch.val_mask[:batch.batch_size]
            test_mask_seed = batch.test_mask[:batch.batch_size]
            y_seed = batch.y[:batch.batch_size]
            
            # 分別收集 Val 和 Test 的預測結果與真實標籤
            val_probs_list.append(out_seed[val_mask_seed].cpu())
            test_probs_list.append(out_seed[test_mask_seed].cpu())
            val_y_list.append(y_seed[val_mask_seed].cpu())
            test_y_list.append(y_seed[test_mask_seed].cpu())

    val_probs = torch.cat(val_probs_list).numpy()
    test_probs = torch.cat(test_probs_list).numpy()
    val_y = torch.cat(val_y_list).numpy()
    test_y = torch.cat(test_y_list).numpy()

    val_preds = (val_probs > threshold).astype(int)
    test_preds = (test_probs > threshold).astype(int)

    return val_probs, val_preds, test_probs, test_preds, val_y, test_y


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
   


# ==============================================================================
# Main - Execute complete GNN anomaly detection  
# ==============================================================================
def main(dataset_name):
    print("=" * 60)
    print("GNN TPE - Optuna")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    # ========================================================================
    # 1. Load Elliptic dataset
    # ========================================================================
    
    # data = load_elliptic_data()
    if dataset_name == 'e1':
        data = dataset_zoo.load_elliptic_data()
    elif dataset_name == 'e2':
        data = dataset_zoo.load_ethereum_data()

    print(f"Graph loaded on: {data.x.device} (Should be CPU)")
    print(f"Batch Size: {batch_size}")

    # ========================================================================
    # 4.1. TPE - Optuna - done
    # ========================================================================
    print("\n4.1 TPE - Optuna GNN models...")
    gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)

    for model_name, _ in gnn_models_list:
        print(model_name, "="*60)
        study = optuna.create_study(direction='maximize', load_if_exists=False, pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        study.optimize(lambda trial: gnn_objective(dataset_name, trial, model_name, data, batch_size, device), n_trials=n_trials)
        
        best_trial = study.best_trial
        tpe_result = {"c1_f1": best_trial.value,"best_params": best_trial.params,}
        print("best param", model_name, "="*60, "\n", tpe_result)

        with open(f"best_model_params/{model_name}_params_{dataset_name}.json", "w") as f:
            json.dump(tpe_result, f, indent=4)
        
        print(f"saved {model_name}_tpe_params_{batch_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: e1 or e2')
    args = parser.parse_args()
    main(args.dataset)