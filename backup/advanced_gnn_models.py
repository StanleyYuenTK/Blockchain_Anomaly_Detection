"""
高級 GNN 模型實現：SGAT、時序特徵提取、半監督學習和集成學習
用於區塊鏈異常檢測，整合多種技術提升檢測能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops
import pandas as pd
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class SpatialTemporalAttention(nn.Module):
    """
    空間-時序注意力機制 (STA)
    用於增強 GNN 對鄰居信息的注意力，特別是多跳鄰居信息
    """
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.5):
        super(SpatialTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        
        # 多頭注意力權重
        self.W_q = nn.Linear(in_channels, hidden_channels)
        self.W_k = nn.Linear(in_channels, hidden_channels)
        self.W_v = nn.Linear(in_channels, hidden_channels)
        
        # 時序注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, timesteps=None):
        """
        Args:
            x: 節點特徵 [num_nodes, in_channels]
            edge_index: 邊索引 [2, num_edges]
            timesteps: 時間步信息 [num_nodes]
        """
        batch_size = x.size(0)
        
        # 空間注意力：基於圖結構的注意力
        Q = self.W_q(x)  # [num_nodes, hidden_channels]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 計算注意力分數（簡化版本，實際應該考慮邊結構）
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / np.sqrt(self.head_dim)
        
        # 使用邊索引構建注意力掩碼
        if edge_index.size(1) > 0:
            # 構建鄰接矩陣掩碼
            adj_mask = torch.zeros(batch_size, batch_size, device=x.device)
            adj_mask[edge_index[0], edge_index[1]] = 1.0
            adj_mask = adj_mask + torch.eye(batch_size, device=x.device)  # 添加自環
            
            # 應用掩碼
            attention_scores = attention_scores.masked_fill(adj_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        spatial_output = torch.matmul(attention_weights, V)
        
        # 時序注意力（如果有時間步信息）
        if timesteps is not None:
            # 按時間步分組處理
            unique_timesteps = torch.unique(timesteps)
            temporal_output = torch.zeros_like(spatial_output)
            
            for t in unique_timesteps:
                mask = (timesteps == t)
                if mask.sum() > 0:
                    # 重塑為序列格式 [1, num_nodes_at_t, hidden_channels]
                    temporal_input = spatial_output[mask].unsqueeze(0)
                    temp_out, _ = self.temporal_attention(temporal_input, temporal_input, temporal_input)
                    temporal_output[mask] = temp_out.squeeze(0)
        else:
            temporal_output = spatial_output
        
        # 殘差連接和層歸一化
        output = self.layer_norm(spatial_output + temporal_output)
        
        return output


class SGATModel(nn.Module):
    """
    SGAT (Spatial-Temporal Attention + GAT) 模型
    整合 STA 和 GAT 的輸出，充分利用多跳鄰居信息
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_heads=8, num_layers=2, dropout=0.5):
        super(SGATModel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        
        # STA 模塊
        self.sta = SpatialTemporalAttention(in_channels, hidden_channels, num_heads, dropout)
        
        # GAT 模塊
        self.gat_convs = nn.ModuleList()
        self.gat_convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, 
                                      dropout=dropout, concat=True))
        
        for _ in range(num_layers - 2):
            self.gat_convs.append(GATConv(hidden_channels * num_heads, hidden_channels, 
                                         heads=num_heads, dropout=dropout, concat=True))
        
        if num_layers > 1:
            self.gat_convs.append(GATConv(hidden_channels * num_heads, hidden_channels, 
                                         heads=1, dropout=dropout, concat=False))
        else:
            self.gat_convs.append(GATConv(in_channels, hidden_channels, 
                                         heads=1, dropout=dropout, concat=False))
        
        # 融合層：整合 STA 和 GAT 的輸出
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, timesteps=None, return_features=False):
        """
        前向傳播
        
        Args:
            x: 節點特徵
            edge_index: 邊索引
            timesteps: 時間步信息
            return_features: 是否返回中間特徵
        """
        # STA 分支
        sta_output = self.sta(x, edge_index, timesteps)
        
        # GAT 分支
        gat_output = x
        for i, conv in enumerate(self.gat_convs[:-1]):
            gat_output = conv(gat_output, edge_index)
            gat_output = F.elu(gat_output)
            gat_output = F.dropout(gat_output, p=self.dropout, training=self.training)
        
        gat_output = self.gat_convs[-1](gat_output, edge_index)
        
        # 融合 STA 和 GAT 的輸出
        combined = torch.cat([sta_output, gat_output], dim=1)
        output = self.fusion(combined)
        
        if return_features:
            return F.log_softmax(output, dim=1), [sta_output, gat_output]
        
        return F.log_softmax(output, dim=1)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE 編碼器
    用於聚合鄰居特徵並生成節點嵌入，重建交易圖
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, aggr='mean'):
        super(GraphSAGEEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """生成節點嵌入"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class TemporalFeatureExtractor(nn.Module):
    """
    時序特徵提取器
    使用 GRU-MHA 和 Conv1D 捕獲時序特徵
    """
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.5):
        super(TemporalFeatureExtractor, self).__init__()
        
        # GRU 層
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
            bidirectional=True
        )
        
        # 多頭注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_channels * 2,  # 雙向 GRU
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Conv1D 層
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, timesteps):
        """
        提取時序特徵
        
        Args:
            x: 節點特徵 [num_nodes, feature_dim]
            timesteps: 時間步 [num_nodes]
        """
        # 按時間步分組
        unique_timesteps = torch.sort(torch.unique(timesteps))[0]
        temporal_features = []
        
        for t in unique_timesteps:
            mask = (timesteps == t)
            if mask.sum() > 0:
                # 獲取該時間步的節點特徵
                t_features = x[mask]  # [num_nodes_at_t, feature_dim]
                
                # GRU 處理（需要序列長度，這裡使用特徵維度作為序列）
                # 重塑為 [1, num_nodes_at_t, feature_dim]
                gru_input = t_features.unsqueeze(0)
                gru_out, _ = self.gru(gru_input)  # [1, num_nodes_at_t, hidden*2]
                
                # 多頭注意力
                mha_out, _ = self.mha(gru_out, gru_out, gru_out)  # [1, num_nodes_at_t, hidden*2]
                
                # Conv1D 處理（需要 [batch, channels, length]）
                conv_input = mha_out.transpose(1, 2)  # [1, hidden*2, num_nodes_at_t]
                conv_out = self.conv1d(conv_input)  # [1, hidden, num_nodes_at_t]
                conv_out = conv_out.transpose(1, 2).squeeze(0)  # [num_nodes_at_t, hidden]
                
                temporal_features.append(conv_out)
        
        # 合併所有時間步的特徵
        if temporal_features:
            output = torch.cat(temporal_features, dim=0)
            # 確保輸出順序與輸入一致
            output_full = torch.zeros(x.size(0), output.size(1), device=x.device)
            idx = 0
            for t in unique_timesteps:
                mask = (timesteps == t)
                n_nodes = mask.sum().item()
                if n_nodes > 0:
                    output_full[mask] = output[idx:idx+n_nodes]
                    idx += n_nodes
            return output_full
        else:
            return x


class HybridModel(nn.Module):
    """
    混合模型：整合 SGAT、GraphSAGE 編碼器和時序特徵提取器
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_heads=8, num_layers=2, dropout=0.5):
        super(HybridModel, self).__init__()
        
        # GraphSAGE 編碼器
        self.sage_encoder = GraphSAGEEncoder(in_channels, hidden_channels, 
                                            hidden_channels, num_layers, dropout)
        
        # 時序特徵提取器
        self.temporal_extractor = TemporalFeatureExtractor(in_channels, hidden_channels, 
                                                           num_heads, dropout)
        
        # SGAT 模型
        self.sgat = SGATModel(hidden_channels * 2, hidden_channels, hidden_channels,
                             num_heads, num_layers, dropout)
        
        # 最終分類層
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, timesteps=None, return_features=False):
        """
        前向傳播
        
        Args:
            x: 節點特徵
            edge_index: 邊索引
            timesteps: 時間步信息
            return_features: 是否返回中間特徵
        """
        # GraphSAGE 編碼
        sage_features = self.sage_encoder(x, edge_index)
        
        # 時序特徵提取
        temporal_features = self.temporal_extractor(x, timesteps)
        
        # 合併特徵
        combined_features = torch.cat([sage_features, temporal_features], dim=1)
        
        # SGAT 處理
        if return_features:
            sgat_output, sgat_features = self.sgat(combined_features, edge_index, timesteps, return_features=True)
            # sgat_features 是 [sta_output, gat_output] 列表
            # 最終分類
            final_features = torch.cat([sgat_features[0], sgat_features[1]], dim=1)
            output = self.classifier(final_features)
            
            return F.log_softmax(output, dim=1), {
                'sage': sage_features,
                'temporal': temporal_features,
                'sgat_sta': sgat_features[0],
                'sgat_gat': sgat_features[1]
            }
        else:
            # 直接獲取 SGAT 的隱藏層特徵用於分類
            sta_output = self.sgat.sta(combined_features, edge_index, timesteps)
            
            gat_output = combined_features
            for i, conv in enumerate(self.sgat.gat_convs[:-1]):
                gat_output = conv(gat_output, edge_index)
                gat_output = F.elu(gat_output)
                gat_output = F.dropout(gat_output, p=self.sgat.dropout, training=self.training)
            gat_output = self.sgat.gat_convs[-1](gat_output, edge_index)
            
            # 最終分類
            final_features = torch.cat([sta_output, gat_output], dim=1)
            output = self.classifier(final_features)
            return F.log_softmax(output, dim=1)


class SelfTraining:
    """
    自訓練半監督學習算法
    用於生成可靠的偽標籤並擴展訓練集
    """
    def __init__(self, model, threshold=0.9, max_iter=5):
        self.model = model
        self.threshold = threshold
        self.max_iter = max_iter
        
    def generate_pseudo_labels(self, data, device):
        """生成偽標籤（僅對未標記節點）"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(data.x, data.edge_index, data.timesteps)
            probs = torch.exp(output)
            max_probs, pred_labels = torch.max(probs, dim=1)
            
            # 只對未標記的節點（y == -1）生成偽標籤
            unlabeled_mask = (data.y == -1)
            
            # 選擇高置信度的預測作為偽標籤
            confident_mask = (max_probs >= self.threshold) & unlabeled_mask
            pseudo_labels = pred_labels[confident_mask]
            pseudo_indices = torch.where(confident_mask)[0]
            
        return pseudo_indices, pseudo_labels
    
    def train_with_pseudo_labels(self, data, optimizer, criterion, device):
        """使用偽標籤進行訓練"""
        for iteration in range(self.max_iter):
            # 生成偽標籤
            pseudo_indices, pseudo_labels = self.generate_pseudo_labels(data, device)
            
            if len(pseudo_indices) == 0:
                print(f"迭代 {iteration+1}: 沒有生成新的偽標籤")
                break
            
            # 擴展訓練集（將偽標籤添加到數據中）
            extended_train_mask = data.train_mask.clone()
            extended_train_mask[pseudo_indices] = True
            
            # 臨時更新標籤（僅用於訓練）
            extended_y = data.y.clone()
            extended_y[pseudo_indices] = pseudo_labels
            
            # 訓練模型
            self.model.train()
            optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.timesteps)
            
            # 計算損失（原始標籤 + 偽標籤）
            loss = criterion(output[extended_train_mask], extended_y[extended_train_mask])
            loss.backward()
            optimizer.step()
            
            print(f"迭代 {iteration+1}: 添加了 {len(pseudo_indices)} 個偽標籤，損失: {loss.item():.4f}")


class SimKernelKMeans:
    """
    相似度核 K-Means 聚類
    用於半監督學習和偽標籤生成
    """
    def __init__(self, n_clusters=2, kernel='rbf', gamma=1.0):
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.gamma = gamma
        self.kmeans = None
        
    def rbf_kernel(self, X, Y=None):
        """RBF 核函數"""
        if Y is None:
            Y = X
        pairwise_dists = torch.cdist(X, Y) ** 2
        return torch.exp(-self.gamma * pairwise_dists)
    
    def fit_predict(self, features):
        """
        使用核 K-Means 進行聚類
        
        Args:
            features: 節點特徵 [num_nodes, feature_dim]
        
        Returns:
            聚類標籤
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # 計算核矩陣
        if self.kernel == 'rbf':
            # 使用 RBF 核的近似方法
            from sklearn.metrics.pairwise import rbf_kernel
            kernel_matrix = rbf_kernel(features, gamma=self.gamma)
        else:
            kernel_matrix = features @ features.T
        
        # 使用 K-Means 在核空間中聚類
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(kernel_matrix)
        
        return labels


class EnsembleModel:
    """
    集成學習框架
    整合多個基分類器的預測結果
    """
    def __init__(self, models, voting='soft'):
        """
        Args:
            models: 基分類器列表
            voting: 'soft' 或 'hard' 投票
        """
        self.models = models
        self.voting = voting
        
    def predict(self, data, device):
        """集成預測"""
        all_predictions = []
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.timesteps)
                probs = torch.exp(output)
                preds = output.argmax(dim=1)
                
                all_predictions.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        if self.voting == 'soft':
            # 軟投票：平均概率
            avg_probs = np.mean(all_probs, axis=0)
            ensemble_pred = np.argmax(avg_probs, axis=1)
        else:
            # 硬投票：多數投票
            all_predictions = np.array(all_predictions)
            ensemble_pred = []
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                ensemble_pred.append(Counter(votes).most_common(1)[0][0])
            ensemble_pred = np.array(ensemble_pred)
        
        return ensemble_pred


def load_elliptic_data_for_advanced(dataset_dir='../Dataset'):
    """加載 Elliptic 數據集（用於高級模型）"""
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')
    
    if not all(os.path.exists(p) for p in [classes_path, edgelist_path, features_path]):
        raise FileNotFoundError(f"找不到數據文件，請確認 Dataset 文件夾中包含所需的 CSV 文件")
    
    print("正在加載數據...")
    
    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)
    
    features_df.rename(columns={0: 'txId'}, inplace=True)
    nodes_df = pd.merge(features_df, classes_df, on='txId', how='left')
    
    feature_columns = nodes_df.columns[2:-1]
    features = nodes_df[feature_columns].values
    x = torch.tensor(features, dtype=torch.float)
    
    labels = nodes_df['class'].apply(lambda c: 1 if c == '2' else (0 if c == '1' else -1))
    y = torch.tensor(labels.values, dtype=torch.long)
    
    tx_ids = nodes_df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}
    
    source_indices = []
    target_indices = []
    for _, row in edgelist_df.iterrows():
        src = row['txId1'] if 'txId1' in edgelist_df.columns else row.iloc[0]
        tgt = row['txId2'] if 'txId2' in edgelist_df.columns else row.iloc[1]
        if src in tx_id_map and tgt in tx_id_map:
            source_indices.append(tx_id_map[src])
            target_indices.append(tx_id_map[tgt])
    
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    timesteps = torch.tensor(nodes_df.iloc[:, 1].values, dtype=torch.long)
    
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps
    
    known_mask = y != -1
    train_mask = (timesteps < 35) & known_mask
    val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    test_mask = (timesteps >= 42) & known_mask
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"數據加載完成！")
    print(f"節點數: {data.num_nodes}, 邊數: {data.num_edges}")
    print(f"訓練節點數: {train_mask.sum().item()}")
    print(f"驗證節點數: {val_mask.sum().item()}")
    print(f"測試節點數: {test_mask.sum().item()}")
    
    return data


def train_advanced_model(model_name='Hybrid', dataset_dir='../Dataset',
                         hidden_channels=128, num_layers=2,
                         learning_rate=0.01, weight_decay=5e-4,
                         epochs=100, dropout=0.5, num_heads=8,
                         use_self_training=False, use_ensemble=False):
    """
    訓練高級模型
    
    Args:
        model_name: 模型名稱 ('SGAT', 'Hybrid')
        dataset_dir: 數據集路徑
        hidden_channels: 隱藏層維度
        num_layers: 層數
        learning_rate: 學習率
        weight_decay: 權重衰減
        epochs: 訓練輪數
        dropout: Dropout 比率
        num_heads: 注意力頭數
        use_self_training: 是否使用自訓練
        use_ensemble: 是否使用集成學習
    """
    # 加載數據
    data = load_elliptic_data_for_advanced(dataset_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建模型
    in_channels = data.num_node_features
    out_channels = 2
    
    if model_name == 'SGAT':
        model = SGATModel(in_channels, hidden_channels, out_channels,
                         num_heads, num_layers, dropout).to(device)
    elif model_name == 'Hybrid':
        model = HybridModel(in_channels, hidden_channels, out_channels,
                           num_heads, num_layers, dropout).to(device)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    print(f"\n{model_name} 模型創建成功")
    print(f"參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    # 自訓練
    if use_self_training:
        print("\n使用自訓練半監督學習...")
        self_trainer = SelfTraining(model, threshold=0.9, max_iter=5)
        self_trainer.train_with_pseudo_labels(data, optimizer, criterion, device)
    
    # 訓練循環
    best_val_f1 = 0
    best_epoch_stats = {}
    
    print(f"\n開始訓練 {model_name} 模型...")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.timesteps)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # 評估
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index, data.timesteps)
            pred = output.argmax(dim=1)
            
            val_f1 = f1_score(data.y[data.val_mask].cpu().numpy(),
                             pred[data.val_mask].cpu().numpy(),
                             average='binary', pos_label=1, zero_division=0)
            val_acc = accuracy_score(data.y[data.val_mask].cpu().numpy(),
                                    pred[data.val_mask].cpu().numpy())
            
            test_f1 = f1_score(data.y[data.test_mask].cpu().numpy(),
                              pred[data.test_mask].cpu().numpy(),
                              average='binary', pos_label=1, zero_division=0)
            test_acc = accuracy_score(data.y[data.test_mask].cpu().numpy(),
                                     pred[data.test_mask].cpu().numpy())
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch_stats = {
                'epoch': epoch,
                'loss': loss.item(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'test_f1': test_f1,
                'test_acc': test_acc
            }
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                  f'Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | '
                  f'Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}')
    
    print("-" * 80)
    print(f"\n{model_name} 訓練完成！")
    print(f"最佳驗證 F1: {best_epoch_stats['val_f1']:.4f} (Epoch {best_epoch_stats['epoch']})")
    print(f"測試 F1: {best_epoch_stats['test_f1']:.4f}")
    print(f"測試準確率: {best_epoch_stats['test_acc']:.4f}")
    
    # 集成學習
    if use_ensemble:
        print("\n使用集成學習...")
        models = [model]  # 可以添加更多模型
        ensemble = EnsembleModel(models, voting='soft')
        ensemble_pred = ensemble.predict(data, device)
        
        ensemble_f1 = f1_score(data.y[data.test_mask].cpu().numpy(),
                              ensemble_pred[data.test_mask.cpu().numpy()],
                              average='binary', pos_label=1, zero_division=0)
        ensemble_acc = accuracy_score(data.y[data.test_mask].cpu().numpy(),
                                     ensemble_pred[data.test_mask.cpu().numpy()])
        
        print(f"集成模型測試 F1: {ensemble_f1:.4f}")
        print(f"集成模型測試準確率: {ensemble_acc:.4f}")
    
    return best_epoch_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練高級 GNN 模型進行區塊鏈異常檢測')
    parser.add_argument('--model', type=str, default='Hybrid', choices=['SGAT', 'Hybrid'],
                       help='選擇模型: SGAT 或 Hybrid')
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
                       help='注意力頭數')
    parser.add_argument('--use_self_training', action='store_true',
                       help='使用自訓練半監督學習')
    parser.add_argument('--use_ensemble', action='store_true',
                       help='使用集成學習')
    
    args = parser.parse_args()
    
    train_advanced_model(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        dropout=args.dropout,
        num_heads=args.num_heads,
        use_self_training=args.use_self_training,
        use_ensemble=args.use_ensemble
    )

