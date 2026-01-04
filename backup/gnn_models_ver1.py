"""
GNN 模型實現：GCN、GAT、GraphSAGE
用於區塊鏈交易分類任務，自動提取圖特徵並進行降維
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) 模型
    用於圖節點分類任務
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # 第一層
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 輸出層
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, out_channels))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 多層 GCN
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最後一層
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) 模型
    使用注意力機制來學習節點之間的關係
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        
        # 第一層：多頭注意力
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        
        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        
        # 輸出層：單頭注意力
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, dropout=dropout, concat=False))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 多層 GAT
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最後一層
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE 模型
    使用採樣和聚合機制來學習節點表示
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # 第一層
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # 輸出層
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 多層 GraphSAGE
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最後一層
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


def load_elliptic_data(dataset_dir='../Dataset'):
    """
    從 Dataset 文件夾加載 Elliptic 數據集
    
    Args:
        dataset_dir (str): Dataset 文件夾的路徑
    
    Returns:
        torch_geometric.data.Data: PyG Data 對象
    """
    # 構建文件路徑
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')
    
    # 檢查文件是否存在
    if not all(os.path.exists(p) for p in [classes_path, edgelist_path, features_path]):
        raise FileNotFoundError(f"找不到數據文件，請確認 Dataset 文件夾中包含所需的 CSV 文件")
    
    print("正在加載數據...")
    
    # 加載數據
    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)
    
    # 重命名特徵數據的第一列為 txId
    features_df.rename(columns={0: 'txId'}, inplace=True)
    
    # 合併特徵和類別標籤
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    
    # 節點特徵：第一列是 txId，第二列是 timestep，最後一列是 class，中間是特徵
    # 特徵從第 3 列開始（索引 2）到倒數第二列
    feature_columns = nodes_df.columns[2:-1]
    features = nodes_df[feature_columns].values
    x = torch.tensor(features, dtype=torch.float)
    
    # 節點標籤：'2' (illicit) -> 1, '1' (licit) -> 0, 'unknown' -> -1
    labels = nodes_df['class'].apply(lambda c: 1 if c == '2' else (0 if c == '1' else -1))
    y = torch.tensor(labels.values, dtype=torch.long)
    
    # 邊索引：將交易 ID 映射到零基索引
    tx_ids = nodes_df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}
    
    # 處理邊列表
    source_indices = []
    target_indices = []
    for _, row in edgelist_df.iterrows():
        src = row['txId1'] if 'txId1' in edgelist_df.columns else row.iloc[0]
        tgt = row['txId2'] if 'txId2' in edgelist_df.columns else row.iloc[1]
        if src in tx_id_map and tgt in tx_id_map:
            source_indices.append(tx_id_map[src])
            target_indices.append(tx_id_map[tgt])
    
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    
    # 時間步用於數據分割
    timesteps = torch.tensor(nodes_df.iloc[:, 1].values, dtype=torch.long)
    
    # 創建 Data 對象
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps
    
    # 創建訓練/驗證/測試掩碼
    # 前 34 個時間步用於訓練，35-41 用於驗證，42-49 用於測試
    known_mask = y != -1
    train_mask = (timesteps < 35) & known_mask
    val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    test_mask = (timesteps >= 42) & known_mask
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"數據加載完成！")
    print(f"節點數: {data.num_nodes}")
    print(f"邊數: {data.num_edges}")
    print(f"特徵維度: {data.num_node_features}")
    print(f"訓練節點數: {train_mask.sum().item()}")
    print(f"驗證節點數: {val_mask.sum().item()}")
    print(f"測試節點數: {test_mask.sum().item()}")
    
    return data


def train_model(model, data, optimizer, criterion, device):
    """
    訓練模型一個 epoch
    
    Args:
        model: GNN 模型
        data: 圖數據
        optimizer: 優化器
        criterion: 損失函數
        device: 設備 (CPU/GPU)
    
    Returns:
        float: 訓練損失
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_model(model, data, device):
    """
    評估模型性能
    
    Args:
        model: GNN 模型
        data: 圖數據
        device: 設備 (CPU/GPU)
    
    Returns:
        tuple: (驗證指標, 測試指標) 每個指標包含 (loss, f1, accuracy)
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # 驗證集評估
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
    val_f1 = f1_score(data.y[data.val_mask].cpu().numpy(), 
                      pred[data.val_mask].cpu().numpy(), 
                      average='binary', pos_label=1, zero_division=0)
    val_acc = accuracy_score(data.y[data.val_mask].cpu().numpy(), 
                            pred[data.val_mask].cpu().numpy())
    
    # 測試集評估
    test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
    test_f1 = f1_score(data.y[data.test_mask].cpu().numpy(), 
                       pred[data.test_mask].cpu().numpy(), 
                       average='binary', pos_label=1, zero_division=0)
    test_acc = accuracy_score(data.y[data.test_mask].cpu().numpy(), 
                             pred[data.test_mask].cpu().numpy())
    
    return (val_loss, val_f1, val_acc), (test_loss, test_f1, test_acc)


def train_and_evaluate(model_name='GCN', dataset_dir='../Dataset', 
                      hidden_channels=128, num_layers=2, 
                      learning_rate=0.01, weight_decay=5e-4, 
                      epochs=100, dropout=0.5, num_heads=8):
    """
    訓練和評估 GNN 模型
    
    Args:
        model_name (str): 模型名稱 ('GCN', 'GAT', 'GraphSAGE')
        dataset_dir (str): 數據集文件夾路徑
        hidden_channels (int): 隱藏層維度
        num_layers (int): 模型層數
        learning_rate (float): 學習率
        weight_decay (float): 權重衰減
        epochs (int): 訓練輪數
        dropout (float): Dropout 比率
        num_heads (int): GAT 模型的注意力頭數
    
    Returns:
        dict: 最佳模型性能指標
    """
    # 加載數據
    data = load_elliptic_data(dataset_dir)
    
    # 設備設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建模型
    in_channels = data.num_node_features
    out_channels = 2  # 二分類：合法(0) 和 非法(1)
    
    if model_name == 'GCN':
        model = GCNModel(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
    elif model_name == 'GAT':
        model = GATModel(in_channels, hidden_channels, out_channels, num_heads, num_layers, dropout).to(device)
    elif model_name == 'GraphSAGE':
        model = GraphSAGEModel(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
    else:
        raise ValueError(f"不支持的模型名稱: {model_name}。請選擇 'GCN', 'GAT', 或 'GraphSAGE'")
    
    print(f"\n{model_name} 模型創建成功")
    print(f"參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 將數據移到設備
    data = data.to(device)
    
    # 優化器和損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    # 訓練循環
    best_val_f1 = 0
    best_epoch_stats = {}
    
    print(f"\n開始訓練 {model_name} 模型...")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        # 訓練
        loss = train_model(model, data, optimizer, criterion, device)
        
        # 評估
        (val_loss, val_f1, val_acc), (test_loss, test_f1, test_acc) = evaluate_model(model, data, device)
        
        # 記錄最佳結果
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch_stats = {
                'epoch': epoch,
                'loss': loss,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_f1': test_f1,
                'test_acc': test_acc
            }
        
        # 定期打印進度
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
                  f'Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | '
                  f'Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}')
    
    # 最終結果
    print("-" * 80)
    print(f"\n{model_name} 訓練完成！")
    print(f"最佳驗證 F1: {best_epoch_stats['val_f1']:.4f} (Epoch {best_epoch_stats['epoch']})")
    print(f"測試 F1: {best_epoch_stats['test_f1']:.4f}")
    print(f"測試準確率: {best_epoch_stats['test_acc']:.4f}")
    
    return best_epoch_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練 GNN 模型進行區塊鏈交易分類')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSAGE'],
                       help='選擇模型: GCN, GAT, 或 GraphSAGE')
    parser.add_argument('--dataset_dir', type=str, default='../Dataset',
                       help='數據集文件夾路徑')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='隱藏層維度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='模型層數')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='學習率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='權重衰減')
    parser.add_argument('--epochs', type=int, default=100,
                       help='訓練輪數')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout 比率')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='GAT 模型的注意力頭數')
    
    args = parser.parse_args()
    
    # 訓練模型
    train_and_evaluate(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        dropout=args.dropout,
        num_heads=args.num_heads
    )

