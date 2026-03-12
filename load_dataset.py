import os
import pandas as pd
import torch
from torch_geometric.data import Data


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