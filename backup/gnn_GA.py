import gc
import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import json
import optuna
import pygad

# GNN Models
import inspect
import gnn_zoo
import dataset_zoo


## Focal Loss https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
from kornia.losses import FocalLoss
import argparse

RANDOM_SEED = 24027277
n_trials=50
epochs=200

# ==============================================================================
# TPE - Optuna
# ==============================================================================

# ==============================================================================
# Models - GNN Models, Isolation Forest Baseline
# ==============================================================================

# GNN Model
def get_gnn(model_name, data, best_params, device):
    model_func = getattr(gnn_zoo, model_name, None)
    model = model_func(best_params).to(device)
    return model

def make_fitness_func(model_name, data, device):
    data = data.to(device)

    def fitness_func(ga_instance, solution, solution_idx):
        # 解碼基因 (Genes)
        # solution = [hidden_channels, ppr_alpha, fl_gamma, fl_alpha]
        if model_name == "MixHop_Model":
            POWER_OPTIONS = ['(0, 1, 2)','(0, 1, 2, 3)']
            hidden_channels = int(solution[0])
            dropout = solution[1]
            powers = POWER_OPTIONS[int(solution[2])]
            fl_gamma = solution[3]
            fl_alpha = solution[4]

            gnn_best_params = {
                'in_channels':data.x.size(1),
                'out_channels': 2,
                'hidden_channels':hidden_channels,
                'dropout':dropout,
                'powers':powers,
            }
        if model_name == "GIN_Model":

            JK = ['max', 'cat']
            hidden_channels = int(solution[0])
            dropout = solution[1]
            num_layers = int(solution[2])
            jk = JK[int(solution[3])]
            fl_gamma = solution[4]
            fl_alpha = solution[5]

            gnn_best_params = {
                'in_channels':data.x.size(1),
                'out_channels': 2,
                'hidden_channels':hidden_channels,
                'dropout':dropout,
                'num_layers':num_layers,
                'jk':jk,
            }

        # 初始化模型與 Loss
        model = get_gnn(model_name, data, gnn_best_params, device).to(device)
        criterion = FocalLoss(alpha=fl_alpha, gamma=fl_gamma, reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        
        # 快速訓練 (Reduced Epochs for GA efficiency)
        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index) 
            pred = out.argmax(dim=1)
            val_f1 = f1_score(data.y[data.val_mask].cpu(), 
                            pred[data.val_mask].cpu(), 
                            average='macro')
        return val_f1
    return fitness_func
   
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
    data = data.to(device)
    
    # ========================================================================
    # 2. TPE - Optuna - done
    # ========================================================================
    print("\n2. TPE - Optuna GNN models...")
    # gnn_models_list = inspect.getmembers(gnn_zoo, inspect.isfunction)

    # for model_name, _ in gnn_models_list:
    model_name= 'GIN_Model'
    print(model_name, "="*60)
    
    if model_name == "MixHop_Model":
        gene_space = [
            range(16, 129, 16),      # hidden_channels: 16 到 128
            {'low':0.1, 'high':0.6}, #dropout
            [0, 1], #power index
            # {'low': 0.05, 'high': 0.3}, # ppr_alpha: 影響訊息擴散範圍
            {'low': 1.0, 'high': 5.0},  # fl_gamma: 對困難樣本的專注度
            {'low': 0.1, 'high': 0.9}   # fl_alpha: 類別權重平衡
        ]
    elif model_name =="GIN_Model":
        gene_space = [
            range(16, 257, 16),      # hidden_channels: 16 到 128
            {'low':0.1, 'high':0.6}, #dropout
            {'low': 1, 'high': 5},  # num_layers
            [0, 1], #jk
            {'low': 1.0, 'high': 5.0},  # fl_gamma: 對困難樣本的專注度
            {'low': 0.1, 'high': 0.9}   # fl_alpha: 類別權重平衡
        ]


    curr_fitness_func = make_fitness_func(model_name, data, torch.device("cuda"))

    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=5,
        fitness_func=curr_fitness_func,
        sol_per_pop=10,
        num_genes=len(gene_space),
        gene_space=gene_space,
        parent_selection_type="sss", # Steady State Selection
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=25
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"best solution_fitness = {solution_fitness}")
    print(f"best solution_idx = {solution_idx}")
    
    
    # with open(f"best_model_params/{model_name}_params_{dataset_name}.json", "w") as f:
        # json.dump(tpe_result, f, indent=4)
    
    # print(f"saved {model_name}_tpe_params")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blockchain Anomaly Detection with GNNs')
    parser.add_argument('dataset', type=str,help='Pleaese select dataset: e1 or e2')
    args = parser.parse_args()
    main(args.dataset)