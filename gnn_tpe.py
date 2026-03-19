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
import gc
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, confusion_matrix
from torch_geometric.loader import NeighborLoader, ImbalancedSampler

# model
import json

# Optimization
import optuna


# GNN Models
import inspect
import gnn_zoo
import dataset_zoo

## Focal Loss https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
from kornia.losses import FocalLoss
import argparse

RANDOM_SEED = 24027277
w_m_f1, w_c1_f1, w_auc = 0.2, 0.7, 0.1
n_trials=30

# ==============================================================================
# NeighborLoader for mini-batch training
# =============================================================================
def get_train_loader(data, batch_size=512, num_neighbors=[25, 10]):
    sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors, # 採樣層數需對應模型層數
        batch_size=batch_size,
        input_nodes=data.train_mask,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True, # 保持進程不被銷毀，節省重啟開銷
    )

# ==============================================================================
# TPE - Optuna
# ==============================================================================

def gnn_objective(trial, model_name, data, batch_size, num_neighbors, device):
    """Optuna objective function for GNN model optimization"""
    try:
        ## for create model ==============================================================================
        in_channels = data.x.size(1)
        out_channels = 2
        # Define search space 
        hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256]) #建議從 64 或 128 開始。Class 1 樣本極少，過大的隱藏層（如 256）會讓模型「記住」少數正樣本的特徵而非「學習」規律，導致泛化能力下降。
        dropout = trial.suggest_float('dropout', 0.2, 0.6, log=False) # 這是防止過擬合的關鍵。對於不平衡數據，建議維持在 0.3 - 0.5。較高的 Dropout 能強迫模型依賴更多樣化的路徑來學習，提高對少數類別的魯棒性。
        # lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True) # 對於不平衡數據，建議 較小的學習率（如 1e-3 或 5e-4）。過大的 LR 會讓模型在訓練初期就被佔多數的 Class 0 梯度主導，直接忽略 Class 1。
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True) # 對於不平衡數據，建議 較小的學習率（如 1e-3 或 5e-4）。過大的 LR 會讓模型在訓練初期就被佔多數的 Class 0 梯度主導，直接忽略 Class 1。
        
        # for training and testing ==============================================================================
        focalloss_alpha = trial.suggest_categorical('focalloss_alpha', [0.8, 0.85, 0.9, 0.95]) #[0.8, 0.9, 0.95]
        focalloss_gamma = trial.suggest_float('focalloss_gamma', 4.0, 6.0)
        threshold = trial.suggest_float('threshold', 0.4, 0.6, log=False)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        if model_name in ["GCN_Model", "GAT_Model", "GraphSAGE_Model", "GIN_Model"]:
            num_layers = trial.suggest_categorical('num_layers', [2, 3, 4, 5]) #2 或 3 層 通常最優。GNN 存在「過度平滑（Over-smoothing）」問題，層數越多，節點特徵越趨同，這會讓你極難區分 14% 的釣魚節點與 86% 的正常節點。

        if "GAT" in model_name:
            heads = trial.suggest_categorical('heads', [4, 8, 16, 32]) # 8 通常優於 4。多頭注意力機制能讓模型同時關注不同的鄰居特徵（例如一個頭關注金額，另一個關注時間頻率），這對捕捉複雜的詐騙模式極其有效。
        else:
            heads = 0

        if "MixHop" in model_name:
            powers = trial.suggest_categorical('powers', [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]], log=True)
        else:
            powers = [0, 1, 2]

        print(f"current heads:{heads}, powers={powers}")

        # Create GNN model
        gnn_best_params = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'out_channels': out_channels,
            'dropout': dropout,
            'heads': heads,
            'powers':powers
        }
        model = get_gnn(model_name, data, gnn_best_params, device)

        # training model --------------
        # define search space for training, testing parameters
        loader = get_train_loader(data, batch_size, num_neighbors)
        
        # training and testing and evaluate
        model = train_gnn_minibatch(model, loader, device, epochs=100, lr=lr, alpha=focalloss_alpha, gamma=focalloss_gamma, weight_decay=weight_decay)
        # val_probs, val_preds, test_probs, test_preds = test_gnn(model, data, threshold=threshold)
        val_probs, val_preds, test_probs, test_preds, val_y, test_y = test_gnn(
            model, data, batch_size, num_neighbors, device, threshold=threshold
        )
        # model_val_performance, model_test_performance = eval_gnn(
        #     model_name, data.y[data.val_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy(), val_probs, val_preds, test_probs, test_preds
        #     )
        model_val_performance, model_test_performance = eval_gnn(
            model_name, val_y, test_y, val_probs, val_preds, test_probs, test_preds
        )
        ## macro f1 score, class 1 f1-score, auc
        return model_val_performance['macro f1-score'], model_val_performance['class 1 f1-score'], model_val_performance['auc']
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



def train_gnn_minibatch(model, loader, device, epochs=100, lr=0.002, alpha=0.875, gamma=2.0, weight_decay=5e-4):
    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    return model

def test_gnn(model, data, batch_size, num_neighbors, device, threshold=0.25): 
    model.eval()
    inference_nodes = torch.where(data.val_mask | data.test_mask)[0]
    inference_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=inference_nodes,
        num_workers=1,
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
    # print(classification_report(y_test, test_preds, zero_division=0))
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
    if dataset_name == 'elliptic':
        data = dataset_zoo.load_elliptic_data()
        batch_size=1024
        num_neighbors=[25, 10]
    elif dataset_name == 'ethereum':
        data = dataset_zoo.load_ethereum_data()
        batch_size=1024 #131072 2.5min/epoch, 262144 1min/epoch, 524288 2 min/epoch failed
        num_neighbors=[25, 10]
    # data = data.to(device)

    print(f"Graph loaded on: {data.x.device} (Should be CPU)")
    print(f"Batch Size: {batch_size}")

    # ========================================================================
    # 4.1. TPE - Optuna - done
    # ========================================================================
    print("\n4.1 TPE - Optuna GNN models...")
    tpe_results = {}
    gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)
    # print(type(gnn_models_list))

    for model_name, _ in gnn_models_list:
        print(model_name, "="*60)
        study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'], load_if_exists=False)
        study.optimize(lambda trial: gnn_objective(trial, model_name, data, batch_size, num_neighbors, device), n_trials=n_trials)
        
        best_trials = max(study.best_trials, key=lambda t: w_m_f1 * t.values[0] + w_c1_f1 * t.values[1] + w_auc * t.values[2])
        tpe_results[model_name] = {
            "macro_f1": best_trials.values[0],
            "c1_f1": best_trials.values[1],
            "auc": best_trials.values[2],
            "best_params": best_trials.params,
        }
        print("best param", model_name, "="*60, "\n", tpe_results[model_name])

        
        # 【關鍵動作】將搜索結果保存，然後結束程式
        with open(f"/best_model_params/{model_name}_tpe_params_{batch_size}.json", "w") as f:
            json.dump(tpe_results, f, indent=4)
        
        print(f"saved {model_name}_tpe_params_{batch_size}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: elliptic or ethereum')
    args = parser.parse_args()
    main(args.dataset)