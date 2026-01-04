"""
# 訓練 GCN 模型並自動進行特徵提取和降維（使用 PCA）
python gnn_models.py --model GCN --epochs 100 --reduction_method pca

# 訓練 GAT 模型並使用 t-SNE 降維
python gnn_models.py --model GAT --epochs 100 --reduction_method tsne

# 跳過特徵提取和可視化
python gnn_models.py --model GraphSAGE --no_extract_features

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
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安裝，將跳過可視化功能")


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) 模型
    用於圖節點分類任務，支持特徵提取
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
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

    def forward(self, x, edge_index, return_features=False):
        """
        前向傳播
        
        Args:
            x: 節點特徵
            edge_index: 邊索引
            return_features: 是否返回中間層特徵
        
        Returns:
            如果 return_features=True，返回 (log_softmax_output, hidden_features)
            否則只返回 log_softmax_output
        """
        hidden_features = []
        
        # 多層 GCN
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            if return_features:
                hidden_features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最後一層
        x = self.convs[-1](x, edge_index)
        output = F.log_softmax(x, dim=1)
        
        if return_features:
            return output, hidden_features
        return output


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) 模型
    使用注意力機制來學習節點之間的關係，支持特徵提取
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
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

    def forward(self, x, edge_index, return_features=False):
        """
        前向傳播
        
        Args:
            x: 節點特徵
            edge_index: 邊索引
            return_features: 是否返回中間層特徵
        
        Returns:
            如果 return_features=True，返回 (log_softmax_output, hidden_features)
            否則只返回 log_softmax_output
        """
        hidden_features = []
        
        # 多層 GAT
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            if return_features:
                hidden_features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最後一層
        x = self.convs[-1](x, edge_index)
        output = F.log_softmax(x, dim=1)
        
        if return_features:
            return output, hidden_features
        return output


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE 模型
    使用採樣和聚合機制來學習節點表示，支持特徵提取
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
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

    def forward(self, x, edge_index, return_features=False):
        """
        前向傳播
        
        Args:
            x: 節點特徵
            edge_index: 邊索引
            return_features: 是否返回中間層特徵
        
        Returns:
            如果 return_features=True，返回 (log_softmax_output, hidden_features)
            否則只返回 log_softmax_output
        """
        hidden_features = []
        
        # 多層 GraphSAGE
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            if return_features:
                hidden_features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最後一層
        x = self.convs[-1](x, edge_index)
        output = F.log_softmax(x, dim=1)
        
        if return_features:
            return output, hidden_features
        return output


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
def extract_features(model, data, device, layer_idx=-1):
    """
    從訓練好的 GNN 模型中提取節點特徵表示
    
    Args:
        model: 訓練好的 GNN 模型
        data: 圖數據
        device: 設備 (CPU/GPU)
        layer_idx: 要提取的層索引 (-1 表示最後一層隱藏層)
    
    Returns:
        numpy.ndarray: 提取的特徵矩陣 [num_nodes, feature_dim]
    """
    model.eval()
    out, hidden_features = model(data.x, data.edge_index, return_features=True)
    
    if layer_idx == -1:
        # 返回最後一層隱藏層特徵
        features = hidden_features[-1] if hidden_features else data.x
    else:
        # 返回指定層的特徵
        if 0 <= layer_idx < len(hidden_features):
            features = hidden_features[layer_idx]
        else:
            raise ValueError(f"層索引 {layer_idx} 超出範圍 [0, {len(hidden_features)-1}]")
    
    return features.cpu().numpy()


def reduce_dimension_pca(features, n_components=2):
    """
    使用 PCA 進行降維
    
    Args:
        features: 特徵矩陣 [num_nodes, feature_dim]
        n_components: 降維後的維度
    
    Returns:
        numpy.ndarray: 降維後的特徵 [num_nodes, n_components]
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(features)
    print(f"PCA 降維完成: {features.shape[1]} -> {n_components} 維")
    print(f"解釋方差比: {pca.explained_variance_ratio_.sum():.4f}")
    return reduced_features


def reduce_dimension_tsne(features, n_components=2, perplexity=30, n_iter=1000):
    """
    使用 t-SNE 進行降維
    
    Args:
        features: 特徵矩陣 [num_nodes, feature_dim]
        n_components: 降維後的維度
        perplexity: t-SNE 的困惑度參數
        n_iter: 迭代次數
    
    Returns:
        numpy.ndarray: 降維後的特徵 [num_nodes, n_components]
    """
    print(f"開始 t-SNE 降維 (這可能需要一些時間)...")
    # 如果特徵維度太高，先用 PCA 降維到 50 維
    if features.shape[1] > 50:
        print(f"特徵維度 {features.shape[1]} 過高，先使用 PCA 降維到 50 維")
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=42, verbose=1)
    reduced_features = tsne.fit_transform(features)
    print(f"t-SNE 降維完成: {features.shape[1]} -> {n_components} 維")
    return reduced_features


def visualize_features_2d(reduced_features, labels, title="特徵可視化", save_path=None):
    """
    可視化 2D 降維後的特徵
    
    Args:
        reduced_features: 2D 特徵矩陣 [num_nodes, 2]
        labels: 節點標籤
        title: 圖表標題
        save_path: 保存路徑（可選）
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安裝，跳過可視化")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 只顯示已知標籤的節點
    known_mask = labels != -1
    known_features = reduced_features[known_mask]
    known_labels = labels[known_mask]
    
    # 繪製不同類別的點
    for label in [0, 1]:
        mask = known_labels == label
        label_name = "合法" if label == 0 else "非法"
        plt.scatter(known_features[mask, 0], known_features[mask, 1], 
                   label=label_name, alpha=0.6, s=10)
    
    plt.xlabel("第一主成分" if "PCA" in title else "t-SNE 維度 1")
    plt.ylabel("第二主成分" if "PCA" in title else "t-SNE 維度 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存到: {save_path}")
    
    plt.show()


@torch.no_grad()
def evaluate_model(model, data, device, detailed=False):
    """
    評估模型性能，支持詳細分類報告
    
    Args:
        model: GNN 模型
        data: 圖數據
        device: 設備 (CPU/GPU)
        detailed: 是否輸出詳細分類報告
    
    Returns:
        tuple: (驗證指標, 測試指標) 每個指標包含 (loss, f1, accuracy)
        如果 detailed=True，還會返回混淆矩陣
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # 驗證集評估
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
    val_y_true = data.y[data.val_mask].cpu().numpy()
    val_y_pred = pred[data.val_mask].cpu().numpy()
    val_f1 = f1_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
    val_acc = accuracy_score(val_y_true, val_y_pred)
    
    # 測試集評估
    test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
    test_y_true = data.y[data.test_mask].cpu().numpy()
    test_y_pred = pred[data.test_mask].cpu().numpy()
    test_f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
    test_acc = accuracy_score(test_y_true, test_y_pred)
    
    if detailed:
        print("\n" + "="*80)
        print("詳細分類報告 - 驗證集")
        print("="*80)
        print(classification_report(val_y_true, val_y_pred, 
                                   target_names=['合法', '非法'], 
                                   zero_division=0))
        print("\n混淆矩陣 - 驗證集")
        print(confusion_matrix(val_y_true, val_y_pred))
        
        print("\n" + "="*80)
        print("詳細分類報告 - 測試集")
        print("="*80)
        print(classification_report(test_y_true, test_y_pred, 
                                   target_names=['合法', '非法'], 
                                   zero_division=0))
        print("\n混淆矩陣 - 測試集")
        print(confusion_matrix(test_y_true, test_y_pred))
        print("="*80 + "\n")
        
        return (val_loss, val_f1, val_acc, val_y_true, val_y_pred), \
               (test_loss, test_f1, test_acc, test_y_true, test_y_pred)
    
    return (val_loss, val_f1, val_acc), (test_loss, test_f1, test_acc)


def perform_feature_extraction_and_reduction(model, data, device, 
                                            reduction_method='pca', 
                                            n_components=2, 
                                            visualize=True,
                                            save_path=None):
    """
    執行自動特徵提取和降維
    
    Args:
        model: 訓練好的 GNN 模型
        data: 圖數據
        device: 設備 (CPU/GPU)
        reduction_method: 降維方法 ('pca' 或 'tsne')
        n_components: 降維後的維度
        visualize: 是否可視化結果
        save_path: 可視化圖表保存路徑
    
    Returns:
        tuple: (原始特徵, 降維後特徵)
    """
    print("\n" + "="*80)
    print("開始自動特徵提取和降維")
    print("="*80)
    
    # 提取特徵
    print("\n步驟 1: 從 GNN 模型提取節點特徵表示...")
    extracted_features = extract_features(model, data, device, layer_idx=-1)
    print(f"提取的特徵形狀: {extracted_features.shape}")
    
    # 降維
    print(f"\n步驟 2: 使用 {reduction_method.upper()} 進行降維...")
    if reduction_method.lower() == 'pca':
        reduced_features = reduce_dimension_pca(extracted_features, n_components)
    elif reduction_method.lower() == 'tsne':
        reduced_features = reduce_dimension_tsne(extracted_features, n_components)
    else:
        raise ValueError(f"不支持的降維方法: {reduction_method}。請選擇 'pca' 或 'tsne'")
    
    # 可視化
    if visualize and n_components == 2:
        print(f"\n步驟 3: 可視化降維後的特徵...")
        title = f"{reduction_method.upper()} 降維可視化 - {type(model).__name__}"
        visualize_features_2d(reduced_features, data.y.cpu().numpy(), 
                              title=title, save_path=save_path)
    
    print("\n特徵提取和降維完成！")
    print("="*80 + "\n")
    
    return extracted_features, reduced_features


def train_and_evaluate(model_name='GCN', dataset_dir='../Dataset', 
                      hidden_channels=128, num_layers=2, 
                      learning_rate=0.01, weight_decay=5e-4, 
                      epochs=100, dropout=0.5, num_heads=8,
                      extract_features_after_training=True,
                      reduction_method='pca',
                      visualize_features=True):
    """
    訓練和評估 GNN 模型，支持自動特徵提取和降維
    
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
        extract_features_after_training (bool): 訓練後是否進行特徵提取和降維
        reduction_method (str): 降維方法 ('pca' 或 'tsne')
        visualize_features (bool): 是否可視化降維後的特徵
    
    Returns:
        dict: 最佳模型性能指標和提取的特徵
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
    
    # 詳細分類報告
    print("\n生成詳細分類報告...")
    evaluate_model(model, data, device, detailed=True)
    
    # 特徵提取和降維
    extracted_features = None
    reduced_features = None
    if extract_features_after_training:
        extracted_features, reduced_features = perform_feature_extraction_and_reduction(
            model, data, device, 
            reduction_method=reduction_method,
            visualize=visualize_features,
            save_path=f"{model_name.lower()}_features_{reduction_method}.png" if visualize_features else None
        )
        best_epoch_stats['extracted_features'] = extracted_features
        best_epoch_stats['reduced_features'] = reduced_features
    
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
    parser.add_argument('--extract_features', action='store_true', default=True,
                       help='訓練後是否進行特徵提取和降維')
    parser.add_argument('--no_extract_features', dest='extract_features', action='store_false',
                       help='跳過特徵提取和降維')
    parser.add_argument('--reduction_method', type=str, default='pca', choices=['pca', 'tsne'],
                       help='降維方法: pca 或 tsne')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='是否可視化降維後的特徵')
    parser.add_argument('--no_visualize', dest='visualize', action='store_false',
                       help='跳過可視化')
    
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
        num_heads=args.num_heads,
        extract_features_after_training=args.extract_features,
        reduction_method=args.reduction_method,
        visualize_features=args.visualize
    )

