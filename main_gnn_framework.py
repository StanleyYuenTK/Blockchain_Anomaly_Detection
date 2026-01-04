"""
main_gnn_framework.py

合併自 `gnn_models_ver2.py` 與 `advanced_gnn_models.py`：
- 統一 imports
- 集中模型定義 (GCN, GAT, GraphSAGE, SGAT, Hybrid)
- 統一 forward(data, return_embed=False) 介面
- 通用 Trainer，支援 nll / focal loss 切換
- 特徵提取 (PCA / t-SNE) 與視覺化呼叫

使用方式 (簡要):
    python main_gnn_framework.py --model GCN --epochs 100 --reduction_method pca
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
import optuna

# 可視化工具
from visualization_tools import plot_confusion_matrix, plot_feature_visualization_2d, plot_training_curves, TrainingHistory

warnings.filterwarnings('ignore')

HAS_MATPLOTLIB = True

# -----------------------
# Model definitions
# -----------------------

class BaseGNN(nn.Module):
    """提供統一的 forward(self, data, return_embed=False) 介面"""
    def forward(self, data, return_embed=False):
        raise NotImplementedError


class GCNModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, out_channels))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            # 返回最後一層隱藏表示（如果沒有隱藏層，就返回輸入）
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GATModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout, concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, dropout=dropout, concat=False))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class GraphSAGEModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
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

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = features[-1] if features else data.x
            return out, embed
        return out


class PCConv(nn.Module):
    """
    PCConv：基於相似度的鄰居選擇（Similarity-based Selector）卷積層。
    對每個目標節點依據餘弦相似度選擇 top_k 鄰居進行聚合，然後透過線性變換輸出。
    """
    def __init__(self, in_channels, out_channels, top_k=10, bias=True):
        super(PCConv, self).__init__()
        self.top_k = top_k
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E] (src, tgt)
        """
        device = x.device
        src = edge_index[0]
        tgt = edge_index[1]
        # normalize features for cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        # per-edge similarity between src and tgt
        sim = (x_norm[src] * x_norm[tgt]).sum(dim=1)  # [E]

        N = x.size(0)
        E = sim.size(0)
        # prepare aggregation tensor
        agg = torch.zeros_like(x, device=device)

        # group edges by target node and select top_k per target
        # NOTE: naive python loop per unique target (prototype; can be optimized)
        unique_targets = torch.unique(tgt)
        for node in unique_targets:
            mask = (tgt == node)
            if mask.sum() == 0:
                continue
            sims = sim[mask]
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            k = min(self.top_k, sims.numel())
            if k < sims.numel():
                topk_vals, topk_idx = torch.topk(sims, k=k)
                selected_global_idx = idxs[topk_idx]
            else:
                selected_global_idx = idxs
            selected_src = src[selected_global_idx]
            selected_sims = sim[selected_global_idx].unsqueeze(1)
            messages = x[selected_src] * selected_sims
            # sum messages into agg[target]
            # use index_add_ for efficiency
            agg.index_add_(0, node.repeat(messages.size(0)).to(device), messages)

        out = self.lin(agg)
        if self.bias is not None:
            out = out + self.bias
        return F.relu(out)


class PC_GNN_Model(BaseGNN):
    """
    PC-GNN: 使用 PCConv 作為基本卷積單元的簡化模型。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, top_k=10, dropout=0.5):
        super(PC_GNN_Model, self).__init__()
        self.num_layers = num_layers
        self.top_k = top_k
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        # first layer
        self.convs.append(PCConv(in_channels, hidden_channels, top_k=top_k))
        for _ in range(num_layers - 2):
            self.convs.append(PCConv(hidden_channels, hidden_channels, top_k=top_k))
        if num_layers > 1:
            self.convs.append(PCConv(hidden_channels, out_channels, top_k=top_k))
        else:
            self.convs.append(PCConv(in_channels, out_channels, top_k=top_k))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        hidden_features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            hidden_features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = hidden_features[-1] if hidden_features else data.x
            return out, embed
        return out


class GINModel(BaseGNN):
    """
    GIN model using GINConv with a 2-layer MLP as the GIN's nn.
    支援 num_layers 與 dropout，並與 existing framework 的 return_embed=True 相容。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # helper to build 2-layer MLP for GINConv
        def make_mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c)
            )

        # first layer
        self.convs.append(GINConv(make_mlp(in_channels, hidden_channels)))
        # middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels)))
        # output layer
        if num_layers > 1:
            self.convs.append(GINConv(make_mlp(hidden_channels, out_channels)))
        else:
            self.convs.append(GINConv(make_mlp(in_channels, out_channels)))

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        hidden_features = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            hidden_features.append(x.clone())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        out = F.log_softmax(x, dim=1)
        if return_embed:
            embed = hidden_features[-1] if hidden_features else data.x
            return out, embed
        return out


# --- Advanced modules (STA, SGAT, Temporal, Hybrid) ---
class SpatialTemporalAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.5):
        super(SpatialTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        head_dim = max(1, hidden_channels // max(1, num_heads))
        self.W_q = nn.Linear(in_channels, hidden_channels)
        self.W_k = nn.Linear(in_channels, hidden_channels)
        self.W_v = nn.Linear(in_channels, hidden_channels)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, timesteps=None):
        Q = self.W_q(x); K = self.W_k(x); V = self.W_v(x)
        # 簡化的空間注意力（使用全部節點相似度並以邊掩碼過濾）
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / np.sqrt(max(1, Q.size(1)//self.num_heads))
        if edge_index is not None and edge_index.numel() > 0:
            mask = torch.zeros(x.size(0), x.size(0), device=x.device)
            mask[edge_index[0], edge_index[1]] = 1.0
            mask = mask + torch.eye(x.size(0), device=x.device)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        spatial_out = torch.matmul(attn_weights, V)

        if timesteps is not None:
            unique_ts = torch.unique(timesteps)
            temporal_out = torch.zeros_like(spatial_out)
            for t in unique_ts:
                mask_t = (timesteps == t)
                if mask_t.sum() > 0:
                    seq = spatial_out[mask_t].unsqueeze(0)
                    temp_out, _ = self.temporal_attention(seq, seq, seq)
                    temporal_out[mask_t] = temp_out.squeeze(0)
        else:
            temporal_out = spatial_out

        out = self.layer_norm(spatial_out + temporal_out)
        return out


class SGATModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super(SGATModel, self).__init__()
        self.sta = SpatialTemporalAttention(in_channels, hidden_channels, num_heads, dropout)
        self.gat_convs = nn.ModuleList()
        self.gat_convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.gat_convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
        if num_layers > 1:
            self.gat_convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout, concat=False))
        else:
            self.gat_convs.append(GATConv(in_channels, hidden_channels, heads=1, dropout=dropout, concat=False))
        self.fusion = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        timesteps = getattr(data, 'timesteps', None)
        sta_out = self.sta(x, edge_index, timesteps)
        gat_x = x
        gat_feats = []
        for conv in self.gat_convs[:-1]:
            gat_x = conv(gat_x, edge_index)
            gat_x = F.elu(gat_x)
            gat_feats.append(gat_x.clone())
            gat_x = F.dropout(gat_x, p=self.dropout, training=self.training)
        gat_x = self.gat_convs[-1](gat_x, edge_index)
        combined = torch.cat([sta_out, gat_x], dim=1)
        out = self.fusion(combined)
        out = F.log_softmax(out, dim=1)
        if return_embed:
            embed = torch.cat([sta_out, gat_x], dim=1)
            return out, embed
        return out


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, aggr='mean'):
        super(GraphSAGEEncoder, self).__init__()
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
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.5):
        super(TemporalFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout if 2>1 else 0)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_channels * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.conv1d = nn.Sequential(nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, timesteps):
        unique_ts = torch.sort(torch.unique(timesteps))[0]
        temporal_features = []
        for t in unique_ts:
            mask = (timesteps == t)
            if mask.sum() > 0:
                t_features = x[mask]
                gru_in = t_features.unsqueeze(0)
                gru_out, _ = self.gru(gru_in)
                mha_out, _ = self.mha(gru_out, gru_out, gru_out)
                conv_in = mha_out.transpose(1, 2)
                conv_out = self.conv1d(conv_in).transpose(1, 2).squeeze(0)
                temporal_features.append(conv_out)
        if temporal_features:
            output = torch.cat(temporal_features, dim=0)
            output_full = torch.zeros(x.size(0), output.size(1), device=x.device)
            idx = 0
            for t in unique_ts:
                mask = (timesteps == t)
                n = mask.sum().item()
                if n > 0:
                    output_full[mask] = output[idx:idx+n]
                    idx += n
            return output_full
        else:
            return x


class HybridModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=2, dropout=0.5):
        super(HybridModel, self).__init__()
        self.sage_encoder = GraphSAGEEncoder(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.temporal_extractor = TemporalFeatureExtractor(in_channels, hidden_channels, num_heads, dropout)
        self.sgat = SGATModel(hidden_channels * 2, hidden_channels, hidden_channels, num_heads, num_layers, dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels))

    def forward(self, data, return_embed=False):
        x, edge_index = data.x, data.edge_index
        timesteps = getattr(data, 'timesteps', None)
        sage_feat = self.sage_encoder(x, edge_index)
        temporal_feat = self.temporal_extractor(x, timesteps) if timesteps is not None else x
        combined = torch.cat([sage_feat, temporal_feat], dim=1)
        # SGAT 期望輸入維度 hidden*2
        sgat_out = self.sgat.forward(Data(x=combined, edge_index=edge_index, timesteps=timesteps), return_embed=True)
        if isinstance(sgat_out, tuple):
            _, sgat_embed = sgat_out
        else:
            sgat_embed = sgat_out
        final = self.classifier(sgat_embed)
        out = F.log_softmax(final, dim=1)
        if return_embed:
            return out, sgat_embed
        return out


# -----------------------
# Utilities: data loading, feature extraction, reduction
# -----------------------
def load_elliptic_data(dataset_dir='../Dataset'):
    classes_path = os.path.join(dataset_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(dataset_dir, 'elliptic_txs_features.csv')
    if not all(os.path.exists(p) for p in [classes_path, edgelist_path, features_path]):
        raise FileNotFoundError("找不到數據文件，請確認 Dataset 文件夾中包含所需的 CSV 文件")
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
            source_indices.append(tx_id_map[src]); target_indices.append(tx_id_map[tgt])
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    timesteps = torch.tensor(nodes_df.iloc[:, 1].values, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    data.timesteps = timesteps
    known_mask = y != -1
    data.train_mask = (timesteps < 35) & known_mask
    data.val_mask = (timesteps >= 35) & (timesteps < 42) & known_mask
    data.test_mask = (timesteps >= 42) & known_mask
    return data


@torch.no_grad()
def extract_features(model, data, device, layer_idx=-1):
    model.eval()
    out = model(data, return_embed=True)
    if isinstance(out, tuple):
        _, embed = out
    else:
        # 有些模型直接回傳 embed（保險處理）
        embed = out
    if isinstance(embed, dict):
        # Hybrid 可能回傳 dict，取 concatenation 或最後一項
        embed = torch.cat([v for v in embed.values()], dim=1)
    return embed.cpu().numpy()


def reduce_dimension_pca(features, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    return reduced


def reduce_dimension_tsne(features, n_components=2, perplexity=30, n_iter=1000):
    if features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=0)
    reduced = tsne.fit_transform(features)
    return reduced


def perform_feature_extraction_and_reduction(model, data, device, reduction_method='pca', n_components=2, visualize=True, save_path=None):
    extracted = extract_features(model, data, device)
    if reduction_method.lower() == 'pca':
        reduced = reduce_dimension_pca(extracted, n_components)
        method_name = 'PCA'
    else:
        reduced = reduce_dimension_tsne(extracted, n_components)
        method_name = 't-SNE'
    if visualize and n_components == 2:
        plot_feature_visualization_2d(reduced, data.y, title=f"{type(model).__name__} 特徵可視化", method=method_name, save_path=save_path)
    return extracted, reduced


# -----------------------
# Model Explanation using GNNExplainer
# -----------------------
def explain_model(model, data, device, num_samples=10, output_dir='explanations'):
    """
    使用 GNNExplainer 解釋模型預測，針對測試集中預測為非法 (Class 1) 的節點
    產生節點特徵重要性與邊重要性的視覺化。

    Args:
        model: 訓練好的 GNN 模型
        data: 圖數據 (torch_geometric.data.Data)
        device: 計算裝置
        num_samples: 解釋的樣本數量
        output_dir: 輸出目錄
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    from torch_geometric.utils import to_networkx
    import seaborn as sns

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 將模型移至指定裝置
    model = model.to(device)
    data = data.to(device)

    # 獲取測試集預測為 Class 1 (非法) 的節點索引
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

    # 篩選測試集中預測為 Class 1 的節點
    test_pred_class_1 = torch.where((data.test_mask) & (pred == 1))[0]

    if len(test_pred_class_1) == 0:
        print("警告: 測試集中沒有節點被預測為 Class 1 (非法)")
        return

    print(f"找到 {len(test_pred_class_1)} 個測試集中預測為非法 (Class 1) 的節點")
    print(f"將對前 {min(num_samples, len(test_pred_class_1))} 個節點進行解釋")

    # 隨機選擇要解釋的樣本
    sample_indices = test_pred_class_1[torch.randperm(len(test_pred_class_1))[:min(num_samples, len(test_pred_class_1))]]

    # 定義 model_forward 封裝函數：將 (x, edge_index) 轉回 data 物件再輸入模型
    def model_forward(x, edge_index):
        # 創建一個新的 Data 物件，使用傳入的 x 和 edge_index
        temp_data = Data(x=x, edge_index=edge_index, y=data.y, train_mask=data.train_mask,
                        val_mask=data.val_mask, test_mask=data.test_mask)
        # 如果原始 data 有其他屬性，也要複製
        if hasattr(data, 'timesteps'):
            temp_data.timesteps = data.timesteps
        return model(temp_data)

    # 創建 GNNExplainer
    explainer = Explainer(
        model=model_forward,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='classification',
            task_level='node',
            return_type='raw',
        ),
    )

    # 對每個選中的節點進行解釋
    for i, node_idx in enumerate(sample_indices):
        print(f"解釋節點 {node_idx.item()} (樣本 {i+1}/{len(sample_indices)})")

        try:
            # 生成解釋
            explanation = explainer(data.x, data.edge_index, index=node_idx)

            # 獲取節點特徵重要性 (node_mask)
            node_importance = explanation.node_mask
            if node_importance is not None:
                node_importance = node_importance.cpu().numpy().flatten()
            else:
                print(f"警告: 節點 {node_idx.item()} 沒有節點重要性資訊")
                continue

            # 獲取邊重要性 (edge_mask)
            edge_importance = explanation.edge_mask
            if edge_importance is not None:
                edge_importance = edge_importance.cpu().numpy().flatten()
            else:
                edge_importance = np.zeros(data.edge_index.size(1))

            # 1. 節點特徵重要性視覺化
            plt.figure(figsize=(12, 6))

            # 節點特徵重要性直方圖
            plt.subplot(1, 2, 1)
            plt.bar(range(len(node_importance)), node_importance, alpha=0.7, color='skyblue')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance Score')
            plt.title(f'Node Feature Importance\\nNode {node_idx.item()} (Predicted: Illegal)')
            plt.xticks(rotation=45)

            # Top 10 重要特徵
            top_features = np.argsort(node_importance)[-10:]
            plt.subplot(1, 2, 2)
            plt.bar(range(len(top_features)), node_importance[top_features], alpha=0.7, color='lightcoral')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance Score')
            plt.title('Top 10 Important Features')
            plt.xticks(range(len(top_features)), top_features, rotation=45)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/node_{node_idx.item()}_features.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 邊重要性視覺化 - 子圖視覺化
            plt.figure(figsize=(15, 10))

            # 獲取節點的鄰域（限制大小以便視覺化）
            node_neighbors = data.edge_index[1][data.edge_index[0] == node_idx]
            if len(node_neighbors) > 50:  # 如果鄰域太大，只取最重要的邊
                # 篩選與目標節點相連的邊
                connected_edges = torch.where((data.edge_index[0] == node_idx) | (data.edge_index[1] == node_idx))[0]
                if len(connected_edges) > 50:
                    # 取最重要的邊
                    edge_scores = edge_importance[connected_edges]
                    top_edge_indices = connected_edges[torch.topk(edge_scores, min(50, len(connected_edges)))[1]]
                else:
                    top_edge_indices = connected_edges
            else:
                # 篩選與目標節點相連的所有邊
                top_edge_indices = torch.where((data.edge_index[0] == node_idx) | (data.edge_index[1] == node_idx))[0]

            # 獲取相關節點集合
            related_nodes = torch.cat([data.edge_index[0][top_edge_indices], data.edge_index[1][top_edge_indices]]).unique()
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(related_nodes)}
            center_node_idx = node_mapping[node_idx.item()]

            # 創建子圖的邊索引
            sub_edge_index = []
            sub_edge_importance = []
            for edge_idx in top_edge_indices:
                src, dst = data.edge_index[:, edge_idx]
                if src.item() in node_mapping and dst.item() in node_mapping:
                    sub_edge_index.append([node_mapping[src.item()], node_mapping[dst.item()]])
                    sub_edge_importance.append(edge_importance[edge_idx])

            if sub_edge_index:
                sub_edge_index = torch.tensor(sub_edge_index).T

                # 創建 NetworkX 圖
                G = nx.Graph()
                G.add_nodes_from(range(len(related_nodes)))
                G.add_edges_from(sub_edge_index.T.tolist())

                # 設置節點位置
                pos = nx.spring_layout(G, seed=42, k=1)

                # 繪製節點
                node_colors = ['red' if i == center_node_idx else 'lightblue' for i in range(len(related_nodes))]
                node_sizes = [300 if i == center_node_idx else 100 for i in range(len(related_nodes))]

                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)

                # 繪製邊（根據重要性著色）
                if sub_edge_importance:
                    edge_colors = plt.cm.plasma(np.array(sub_edge_importance) / max(sub_edge_importance) if max(sub_edge_importance) > 0 else np.array(sub_edge_importance))
                    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.6)

                    # 添加顏色條
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(sub_edge_importance), vmax=max(sub_edge_importance)))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
                    cbar.set_label('Edge Importance')

                # 添加節點標籤
                labels = {i: str(related_nodes[i].item()) for i in range(len(related_nodes))}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

                plt.title(f'Edge Importance Subgraph\\nNode {node_idx.item()} (Predicted: Illegal)')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/node_{node_idx.item()}_edges.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print(f"警告: 節點 {node_idx.item()} 沒有足夠的邊資料進行視覺化")

        except Exception as e:
            print(f"解釋節點 {node_idx.item()} 時發生錯誤: {e}")
            continue

    print(f"解釋完成！結果已保存至 {output_dir} 目錄")
    print(f"每個節點生成兩個圖片：")
    print(f"  - *_features.png: 節點特徵重要性")
    print(f"  - *_edges.png: 邊重要性子圖")


# -----------------------
# Simple graph augmentation (from future_research_methods.simple_graph_augmentation)
# -----------------------
def simple_graph_augmentation(data, edge_drop_rate: float = 0.1, feature_noise_level: float = 0.01):
    """
    對整張圖做簡單增強：隨機刪除部分邊並對節點特徵加噪音。
    回傳新的 Data 物件（不會破壞原始 data）。
    """
    # 輕量的檢查
    if not hasattr(data, 'edge_index') or not hasattr(data, 'x'):
        raise RuntimeError("data 需包含 edge_index 與 x 屬性以使用 simple_graph_augmentation")

    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    # keep mask 為 True 的邊將被保留
    keep_mask = torch.rand(num_edges, device=edge_index.device) > edge_drop_rate
    new_edge_index = edge_index[:, keep_mask]

    # 對特徵加高斯噪音
    new_x = data.x + feature_noise_level * torch.randn_like(data.x)

    aug = Data(x=new_x, y=data.y, edge_index=new_edge_index)
    # 複製常見屬性
    for attr in ["train_mask", "val_mask", "test_mask", "timesteps"]:
        if hasattr(data, attr):
            setattr(aug, attr, getattr(data, attr))
    return aug


def graph_smote(data, ratio: float = 1.0, k: int = 5, device=None):
    """
    簡單的 Graph-SMOTE 實作（節點級別的 SMOTE）：對訓練集中的非法節點（label==1）採樣插值生成 synthetic 節點，
    並將 synthetic 節點連接到原始節點的鄰居（簡化處理）。

    Args:
        data: PyG Data（包含 x, y, edge_index, train_mask）
        ratio: 新增樣本數 / 原非法樣本數（例如 1.0 表示增加相同數量）
        k: 在少數類中選擇近鄰的數量（這裡採隨機選擇近鄰）
        device: torch device
    Returns:
        aug_data: 含合成節點的新的 Data（原始 data 不被破壞）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = data.x
    y = data.y
    edge_index = data.edge_index

    # 僅在有標籤與訓練 mask 的情況下操作
    if not hasattr(data, 'train_mask'):
        return data

    train_mask = data.train_mask
    minority_idx = torch.where((train_mask) & (y == 1))[0]
    n_minority = minority_idx.size(0)
    if n_minority == 0 or ratio <= 0:
        return data

    n_new = int(n_minority * ratio)
    # 如果要生成的數量為0，直接返回
    if n_new == 0:
        return data

    # 隨機選擇基礎樣本與近鄰
    rng = torch.Generator(device=x.device)
    base_indices = minority_idx[torch.randint(0, n_minority, (n_new,), generator=rng)]
    neighbor_choices = minority_idx

    new_features = []
    new_edges = []
    new_labels = []
    new_timesteps = []

    # 取得 adjacency list 便於連接 synthetic 節點
    src = edge_index[0].cpu().numpy()
    tgt = edge_index[1].cpu().numpy()
    adj = {}
    for s, t in zip(src, tgt):
        adj.setdefault(int(s), set()).add(int(t))
        adj.setdefault(int(t), set()).add(int(s))

    for base in base_indices.cpu().numpy():
        # 隨機選一個 minority neighbor
        neigh = int(neighbor_choices[torch.randint(0, n_minority, (1,), generator=rng)].item())
        alpha = float(torch.rand(1).item())
        feat_i = x[base].cpu().numpy()
        feat_j = x[neigh].cpu().numpy()
        new_feat = torch.tensor(feat_i + alpha * (feat_j - feat_i), dtype=x.dtype, device=device)
        new_features.append(new_feat.unsqueeze(0))
        new_labels.append(torch.tensor([1], dtype=y.dtype, device=device))
        # 合成節點連接到 base 與其鄰居
        neighbors_set = set()
        if int(base) in adj:
            neighbors_set |= adj[int(base)]
        if int(neigh) in adj:
            neighbors_set |= adj[int(neigh)]
        for nb in neighbors_set:
            new_edges.append((nb, -1))  # -1 placeholder for synthetic node id to be filled later
        # timesteps 若存在則採用 base 的 timestep
        if hasattr(data, 'timesteps'):
            new_timesteps.append(torch.tensor([int(getattr(data, 'timesteps')[base].item())], device=device))
        else:
            new_timesteps.append(torch.tensor([0], device=device))

    # 合併新節點
    new_x = torch.cat(new_features, dim=0).to(device)
    new_y = torch.cat(new_labels, dim=0).to(device)

    x_cat = torch.cat([x.to(device), new_x], dim=0)
    y_cat = torch.cat([y.to(device), new_y], dim=0)

    # 更新 edge_index：替換 placeholder -1 為真實新節點索引
    orig_n = x.size(0)
    edge_list = edge_index.clone().cpu().tolist()
    new_edge_pairs = []
    # Enumerate synthetic nodes and assign ids
    synth_ids = list(range(orig_n, orig_n + len(new_features)))
    idx = 0
    for pair in new_edges:
        nb, _ = pair
        synth_id = synth_ids[idx // max(1, len(neighbors_set))] if len(synth_ids) > 0 else orig_n
        # above indexing ensures mapping but is approximate; instead map by sequence:
        # We'll assign sequentially: every group of neighbors corresponds to one synthetic node in order
        idx += 1
    # Better approach: rebuild edges by iterating per synthetic node
    new_edge_pairs = []
    for s_idx, base in enumerate(base_indices.cpu().numpy()):
        # synthetic node id
        sid = orig_n + s_idx
        neighbors_set = set()
        if int(base) in adj:
            neighbors_set |= adj[int(base)]
        neigh = int(neighbor_choices[torch.randint(0, n_minority, (1,), generator=rng)].item())
        if int(neigh) in adj:
            neighbors_set |= adj[int(neigh)]
        for nb in neighbors_set:
            new_edge_pairs.append((nb, sid))
            new_edge_pairs.append((sid, nb))

    if len(new_edge_pairs) > 0:
        new_edge_tensor = torch.tensor(new_edge_pairs, dtype=torch.long, device=device).t()
        edge_index_cat = torch.cat([edge_index.to(device), new_edge_tensor], dim=1)
    else:
        edge_index_cat = edge_index.to(device)

    # 更新 masks：將 synthetic 節點標為 train=True
    train_mask_cat = torch.cat([data.train_mask.to(device), torch.ones(len(new_features), dtype=torch.bool, device=device)], dim=0)
    val_mask_cat = torch.cat([data.val_mask.to(device), torch.zeros(len(new_features), dtype=torch.bool, device=device)], dim=0)
    test_mask_cat = torch.cat([data.test_mask.to(device), torch.zeros(len(new_features), dtype=torch.bool, device=device)], dim=0)

    aug = Data(x=x_cat, y=y_cat, edge_index=edge_index_cat)
    aug.train_mask = train_mask_cat
    aug.val_mask = val_mask_cat
    aug.test_mask = test_mask_cat
    if hasattr(data, 'timesteps'):
        aug.timesteps = torch.cat([data.timesteps.to(device), torch.cat(new_timesteps, dim=0)], dim=0)
    return aug


# -----------------------
# Self-training (半監督) 與 EnsembleModel (集成) 來自 advanced_gnn_models.py
class SelfTraining:
    """
    自訓練半監督學習算法。
    用於生成偽標籤並擴展訓練集以改善少標籤場景。
    """
    def __init__(self, model, threshold=0.9, max_iter=5):
        self.model = model
        self.threshold = threshold
        self.max_iter = max_iter

    def generate_pseudo_labels(self, data, device):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            probs = torch.exp(output)
            max_probs, pred_labels = torch.max(probs, dim=1)
            unlabeled_mask = (data.y == -1)
            confident_mask = (max_probs >= self.threshold) & unlabeled_mask
            pseudo_labels = pred_labels[confident_mask]
            pseudo_indices = torch.where(confident_mask)[0]
        return pseudo_indices, pseudo_labels

    def train_with_pseudo_labels(self, data, optimizer, criterion, device):
        for iteration in range(self.max_iter):
            pseudo_indices, pseudo_labels = self.generate_pseudo_labels(data, device)
            if len(pseudo_indices) == 0:
                print(f"SelfTraining: 迭代 {iteration+1} 沒有生成偽標籤，停止")
                break
            extended_train_mask = data.train_mask.clone()
            extended_train_mask[pseudo_indices] = True
            extended_y = data.y.clone()
            extended_y[pseudo_indices] = pseudo_labels
            # 訓練一步
            self.model.train()
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output[extended_train_mask], extended_y[extended_train_mask])
            loss.backward()
            optimizer.step()
            print(f"SelfTraining 迭代 {iteration+1}: 新增偽標籤 {len(pseudo_indices)}，損失 {loss.item():.4f}")


class EnsembleModel:
    """
    集成學習框架 (soft / hard voting)
    """
    def __init__(self, models, voting='soft'):
        self.models = models
        self.voting = voting

    def predict(self, data, device):
        all_predictions = []
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(data)
                probs = torch.exp(output)
                preds = output.argmax(dim=1)
                all_predictions.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        if self.voting == 'soft':
            avg_probs = np.mean(all_probs, axis=0)
            ensemble_pred = np.argmax(avg_probs, axis=1)
        else:
            all_predictions = np.array(all_predictions)
            ensemble_pred = []
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                ensemble_pred.append(Counter(votes).most_common(1)[0][0])
            ensemble_pred = np.array(ensemble_pred)
        return ensemble_pred


# -----------------------
# Losses
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.nll_loss(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# -----------------------
# GAN for minority augmentation (WGAN-GP)
# -----------------------
class GeneratorMLP(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_dim=128):
        super(GeneratorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class DiscriminatorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(DiscriminatorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gradient_penalty(discriminator, real_samples, fake_samples, device, lambda_gp=10.0):
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)
    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return gp


def pretrain_gan_and_augment(data, device, gan_epochs=100, batch_size=64, lr=1e-4, latent_dim=32, augment_ratio=1.0):
    """
    預訓練 GAN 並生成額外的非法樣本加入訓練集。

    Args:
        data: PyG Data 物件
        device: torch device
        gan_epochs: GAN 訓練的 epoch 數
        batch_size: 訓練批次大小
        lr: 學習率
        latent_dim: 噪聲維度
        augment_ratio: 生成樣本數相對於原始少數類的比例

    Returns:
        augmented_data: 包含合成樣本的新的 Data 物件
    """
    print(f"開始預訓練 GAN (epochs={gan_epochs}, batch_size={batch_size}, lr={lr}, latent_dim={latent_dim})")

    # 獲取少數類樣本 (類別 1 - 非法)
    train_mask = data.train_mask
    minority_mask = (train_mask) & (data.y == 1)
    minority_features = data.x[minority_mask].to(device)
    n_minority = minority_features.size(0)
    n_generated = int(n_minority * augment_ratio)

    print(f"原始少數類樣本數: {n_minority}, 將生成 {n_generated} 個合成樣本")

    if n_minority == 0:
        print("警告: 沒有找到少數類樣本，跳過 GAN 預訓練")
        return data

    # 實例化 GAN 組件
    feat_dim = data.num_node_features
    generator = GeneratorMLP(latent_dim, feat_dim, hidden_dim=max(128, feat_dim)).to(device)
    discriminator = DiscriminatorMLP(feat_dim, hidden_dim=max(128, feat_dim)).to(device)

    # 優化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # GAN 訓練循環
    for epoch in range(gan_epochs):
        # 訓練 discriminator
        d_optimizer.zero_grad()

        # 真實樣本
        real_batch_idx = torch.randint(0, n_minority, (min(batch_size, n_minority),), device=device)
        real_batch = minority_features[real_batch_idx]

        # 生成假樣本
        z = torch.randn(real_batch.size(0), latent_dim, device=device)
        fake_batch = generator(z).detach()

        # Discriminator loss
        d_real = discriminator(real_batch)
        d_fake = discriminator(fake_batch)
        gp = gradient_penalty(discriminator, real_batch, fake_batch, device, lambda_gp=10.0)
        d_loss = d_fake.mean() - d_real.mean() + gp

        d_loss.backward()
        d_optimizer.step()

        # 訓練 generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = generator(z)
        g_loss = -discriminator(fake_batch).mean()

        g_loss.backward()
        g_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"GAN Epoch {epoch+1:03d}/{gan_epochs:03d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    print("GAN 預訓練完成")

    # 使用訓練好的生成器生成合成樣本
    augmented_data = gan_augment(data, generator, n_generated, latent_dim=latent_dim, device=device)
    print(f"成功生成 {n_generated} 個合成樣本並加入訓練集")
    print(f"原始數據節點數: {data.x.size(0)}, 擴增後節點數: {augmented_data.x.size(0)}")

    return augmented_data


def gan_augment(data, generator, num_samples, latent_dim=32, device=None):
    """
    使用 generator 生成 num_samples 個非法樣本特徵並將其加入 data (只產生特徵與標籤，不修改邊)
    回傳 augmented Data 物件（不改變原始 data）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = generator.to(device)
    with torch.no_grad():
        z = torch.randn(num_samples, generator.net[0].in_features, device=device)
        fake_feats = gen(z)
    # concat to existing
    x_cat = torch.cat([data.x.to(device), fake_feats], dim=0)
    y_fake = torch.ones(num_samples, dtype=data.y.dtype, device=device)
    y_cat = torch.cat([data.y.to(device), y_fake], dim=0)
    # extend masks: place generated in train set
    train_mask_cat = torch.cat([data.train_mask.to(device), torch.ones(num_samples, dtype=torch.bool, device=device)], dim=0)
    val_mask_cat = torch.cat([data.val_mask.to(device), torch.zeros(num_samples, dtype=torch.bool, device=device)], dim=0)
    test_mask_cat = torch.cat([data.test_mask.to(device), torch.zeros(num_samples, dtype=torch.bool, device=device)], dim=0)
    aug = Data(x=x_cat, y=y_cat, edge_index=data.edge_index.to(device))
    aug.train_mask = train_mask_cat
    aug.val_mask = val_mask_cat
    aug.test_mask = test_mask_cat
    if hasattr(data, 'timesteps'):
        aug.timesteps = torch.cat([data.timesteps.to(device), torch.zeros(num_samples, dtype=data.timesteps.dtype, device=device)], dim=0)
    return aug


# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, model, data, device, optimizer, criterion, history=None):
        self.model = model
        self.data = data
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = history or TrainingHistory()
        # 額外紀錄非法類別與 macro 指標的歷史（方便繪圖）
        self.metric_history = {
            'val_precision_illicit': [],
            'val_recall_illicit': [],
            'val_macro_recall': [],
            'val_auc': [],
            'val_gmean': [],
            'test_precision_illicit': [],
            'test_recall_illicit': [],
            'test_macro_recall': [],
            'test_auc': [],
            'test_gmean': []
        }
        # GAN related defaults
        self.gan_enabled = False
        self.gan_generator = None
        self.gan_discriminator = None
        self.gan_g_optimizer = None
        self.gan_d_optimizer = None
        self.gan_steps = 0
        self.gan_batch_size = 64
        self.gan_latent_dim = 32
        self.gan_gp_lambda = 10.0
        self.gan_gen_ratio = 1.0  # generated samples relative to minority count

        # PGExplainer related defaults
        self.pg_explainer = None
        self.pg_explainer_trained = False

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        # Label-balanced sampler: 若 trainer 被設定為 label_balance 且有 batch_size，則在 train_mask 中抽取平衡子集計算 loss
        use_balanced = getattr(self, 'label_balance', False)
        batch_size = getattr(self, 'batch_size', None)
        if use_balanced and batch_size and batch_size > 0:
            train_idx = torch.where(self.data.train_mask)[0]
            if train_idx.numel() == 0:
                loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            else:
                ys = self.data.y[train_idx]
                pos_mask = (ys == 1)
                neg_mask = (ys == 0)
                pos_idx = train_idx[pos_mask]
                neg_idx = train_idx[neg_mask]
                half = batch_size // 2
                # sample positives
                if pos_idx.numel() == 0:
                    sampled_pos = pos_idx
                elif pos_idx.numel() >= half:
                    perm = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:half]
                    sampled_pos = pos_idx[perm]
                else:
                    # upsample with replacement
                    perm = torch.randint(0, pos_idx.numel(), (half,), device=pos_idx.device)
                    sampled_pos = pos_idx[perm]
                # sample negatives
                if neg_idx.numel() == 0:
                    sampled_neg = neg_idx
                elif neg_idx.numel() >= half:
                    perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:half]
                    sampled_neg = neg_idx[perm]
                else:
                    perm = torch.randint(0, neg_idx.numel(), (half,), device=neg_idx.device)
                    sampled_neg = neg_idx[perm]
                selected = torch.cat([sampled_pos, sampled_neg], dim=0)
                if selected.numel() == 0:
                    loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
                else:
                    loss = self.criterion(out[selected], self.data.y[selected])
        else:
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, detailed=False):
        self.model.eval()
        out = self.model(self.data)
        pred = out.argmax(dim=1)
        val_loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask]).item()
        val_y_true = self.data.y[self.data.val_mask].cpu().numpy()
        val_y_pred = pred[self.data.val_mask].cpu().numpy()
        val_f1 = f1_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
        val_acc = accuracy_score(val_y_true, val_y_pred)
        # 計算機率以供 AUC 使用（模型輸出為 log_softmax）
        probs = torch.exp(out).cpu().numpy()
        val_mask_np = self.data.val_mask.cpu().numpy()
        test_mask_np = self.data.test_mask.cpu().numpy()
        try:
            val_recall_macro = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        except Exception:
            val_recall_macro = 0.0
        try:
            val_precision_illicit = precision_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
        except Exception:
            val_precision_illicit = 0.0
        try:
            val_recall_illicit = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
        except Exception:
            val_recall_illicit = 0.0
        try:
            val_auc = roc_auc_score(val_y_true, probs[val_mask_np, 1])
        except Exception:
            # try multi-class safe computation
            try:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(val_y_true)
                y_bin = label_binarize(val_y_true, classes=classes)
                probs_all = probs[val_mask_np]
                val_auc = roc_auc_score(y_bin, probs_all, average='macro', multi_class='ovr')
            except Exception:
                val_auc = float('nan')
        try:
            val_macro_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        except Exception:
            val_macro_f1 = 0.0
        # G-Mean: sqrt(TPR * TNR) where TPR = recall positive, TNR = recall negative (specificity)
        try:
            val_tpr = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
            val_tnr = recall_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
            val_gmean = float((val_tpr * val_tnr) ** 0.5)
        except Exception:
            val_gmean = 0.0

        test_loss = self.criterion(out[self.data.test_mask], self.data.y[self.data.test_mask]).item()
        test_y_true = self.data.y[self.data.test_mask].cpu().numpy()
        test_y_pred = pred[self.data.test_mask].cpu().numpy()
        test_f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
        test_acc = accuracy_score(test_y_true, test_y_pred)
        try:
            test_recall_macro = recall_score(test_y_true, test_y_pred, average='macro', zero_division=0)
        except Exception:
            test_recall_macro = 0.0
        try:
            test_precision_illicit = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
        except Exception:
            test_precision_illicit = 0.0
        try:
            test_recall_illicit = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
        except Exception:
            test_recall_illicit = 0.0
        try:
            test_auc = roc_auc_score(test_y_true, probs[test_mask_np, 1])
        except Exception:
            try:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(test_y_true)
                y_bin = label_binarize(test_y_true, classes=classes)
                probs_all = probs[test_mask_np]
                test_auc = roc_auc_score(y_bin, probs_all, average='macro', multi_class='ovr')
            except Exception:
                test_auc = float('nan')
        try:
            test_macro_f1 = f1_score(test_y_true, test_y_pred, average='macro', zero_division=0)
        except Exception:
            test_macro_f1 = 0.0
        try:
            test_tpr = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_tnr = recall_score(test_y_true, test_y_pred, pos_label=0, zero_division=0)
            test_gmean = float((test_tpr * test_tnr) ** 0.5)
        except Exception:
            test_gmean = 0.0

        if detailed:
            print("驗證集分類報告:")
            print(classification_report(val_y_true, val_y_pred, target_names=['合法', '非法'], zero_division=0))
            print("混淆矩陣 (驗證):")
            print(confusion_matrix(val_y_true, val_y_pred))
            print(f"驗證 Macro Recall: {val_recall_macro:.4f} | 驗證 AUC: {val_auc}")
            print(f"驗證 非法類別 Precision: {val_precision_illicit:.4f} | Recall: {val_recall_illicit:.4f}")
            print("\n測試集分類報告:")
            print(classification_report(test_y_true, test_y_pred, target_names=['合法', '非法'], zero_division=0))
            print("混淆矩陣 (測試):")
            print(confusion_matrix(test_y_true, test_y_pred))
            print(f"測試 Macro Recall: {test_recall_macro:.4f} | 測試 AUC: {test_auc}")
            print(f"測試 非法類別 Precision: {test_precision_illicit:.4f} | Recall: {test_recall_illicit:.4f}")
            return (val_loss, val_f1, val_acc, val_precision_illicit, val_recall_illicit, val_recall_macro, val_auc, val_macro_f1, val_gmean, val_y_true, val_y_pred), (test_loss, test_f1, test_acc, test_precision_illicit, test_recall_illicit, test_recall_macro, test_auc, test_macro_f1, test_gmean, test_y_true, test_y_pred)

        return (val_loss, val_f1, val_acc, val_precision_illicit, val_recall_illicit, val_recall_macro, val_auc, val_macro_f1, val_gmean), (test_loss, test_f1, test_acc, test_precision_illicit, test_recall_illicit, test_recall_macro, test_auc, test_macro_f1, test_gmean)

    def fit(self, epochs=100, log_interval=10, extract_features_after=True, reduction_method='pca', history_save_path=None):
        best_val_f1 = 0
        best_stats = {}
        self.data = self.data.to(self.device)
        self.model = self.model.to(self.device)
        for epoch in range(1, epochs+1):
            # -- GAN training at epoch start (if enabled)
            if getattr(self, 'gan_enabled', False):
                try:
                    gen = self.gan_generator.to(self.device)
                    dis = self.gan_discriminator.to(self.device)
                    g_opt = self.gan_g_optimizer
                    d_opt = self.gan_d_optimizer
                    latent_dim = self.gan_latent_dim
                    gp_lambda = self.gan_gp_lambda
                    batch_size = self.gan_batch_size
                    # collect minority features from training mask
                    train_idx = torch.where(self.data.train_mask)[0]
                    minority_idx = train_idx[self.data.y[train_idx] == 1]
                    if minority_idx.numel() > 0:
                        real_feats = self.data.x[minority_idx].to(self.device)
                        for step in range(max(1, self.gan_steps)):
                            # train discriminator
                            d_opt.zero_grad()
                            # sample real batch
                            idx = torch.randint(0, real_feats.size(0), (min(batch_size, real_feats.size(0)),), device=self.device)
                            real_batch = real_feats[idx]
                            z = torch.randn(real_batch.size(0), latent_dim, device=self.device)
                            fake_batch = gen(z).detach()
                            d_real = dis(real_batch)
                            d_fake = dis(fake_batch)
                            gp = gradient_penalty(dis, real_batch, fake_batch, self.device, lambda_gp=gp_lambda)
                            d_loss = d_fake.mean() - d_real.mean() + gp
                            d_loss.backward()
                            d_opt.step()
                            # train generator
                            g_opt.zero_grad()
                            z = torch.randn(batch_size, latent_dim, device=self.device)
                            fake = gen(z)
                            g_loss = -dis(fake).mean()
                            g_loss.backward()
                            g_opt.step()
                except Exception as e:
                    print(f"[警告] GAN 訓練步驟失敗: {e}")
            # 每個 epoch 前對 data 做 graph augmentation（臨時替換 self.data 進行訓練）
            orig_data = self.data
            # 如果 GAN enabled，生成樣本並合併到 data 以供後續訓練
            if getattr(self, 'gan_enabled', False) and getattr(self, 'gan_generator', None) is not None:
                try:
                    # determine number of generated samples: ratio * minority_count
                    train_idx = torch.where(orig_data.train_mask)[0]
                    minority_idx = train_idx[orig_data.y[train_idx] == 1]
                    n_min = minority_idx.numel()
                    n_gen = int(max(0, getattr(self, 'gan_gen_ratio', 1.0) * n_min))
                    if n_gen > 0:
                        aug_data = gan_augment(orig_data, self.gan_generator, n_gen, latent_dim=self.gan_latent_dim, device=self.device)
                        self.data = aug_data
                        # ensure subsequent augmentation / evaluation uses augmented data
                except Exception as e:
                    print(f"[警告] GAN augment 失敗: {e}")
            try:
                aug_data = simple_graph_augmentation(orig_data)
                # 保持在相同 device
                aug_data = aug_data.to(self.device)
                self.data = aug_data
            except Exception as e:
                print(f"[警告] graph augmentation 失敗，將使用原始資料訓練: {e}")

            train_loss = self.train_epoch()
            # 還原原始資料以便後續評估 / 特徵提取
            self.data = orig_data
            eval_res = self.evaluate(detailed=False)
            (val_loss, val_f1, val_acc, val_prec_illicit, val_rec_illicit, val_recall_macro, val_auc, val_macro_f1, val_gmean), (test_loss, test_f1, test_acc, test_prec_illicit, test_rec_illicit, test_recall_macro, test_auc, test_macro_f1, test_gmean) = eval_res
            # 更新 training history（保持原有 TrainingHistory 結構）
            # 新增 epoch 紀錄，包含 Macro Recall 與 G-Mean
            self.history.add_epoch(epoch, train_loss, val_loss, val_f1, val_acc, test_f1, test_acc,
                                   val_macro_recall=val_recall_macro, test_macro_recall=test_recall_macro,
                                   val_gmean=val_gmean, test_gmean=test_gmean,
                                   val_macro_f1=val_macro_f1, test_macro_f1=test_macro_f1,
                                   val_macro_auc=val_auc, test_macro_auc=test_auc)
            # 更新額外 metric history
            self.metric_history['val_precision_illicit'].append(val_prec_illicit)
            self.metric_history['val_recall_illicit'].append(val_rec_illicit)
            self.metric_history['val_macro_recall'].append(val_recall_macro)
            self.metric_history['val_auc'].append(val_auc)
            self.metric_history['val_gmean'].append(val_gmean)
            self.metric_history['test_precision_illicit'].append(test_prec_illicit)
            self.metric_history['test_recall_illicit'].append(test_rec_illicit)
            self.metric_history['test_macro_recall'].append(test_recall_macro)
            self.metric_history['test_auc'].append(test_auc)
            self.metric_history['test_gmean'].append(test_gmean)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_stats = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_f1': val_f1, 'val_acc': val_acc, 'test_f1': test_f1, 'test_acc': test_acc}
            if epoch % log_interval == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
        print("訓練完成，最佳驗證 F1:", best_val_f1)
        # 詳細報告
        full_val, full_test = self.evaluate(detailed=True)
        # 視覺化
        # 混淆矩陣
        # detailed returns: (..., val_gmean, val_y_true, val_y_pred)
        val_y_true = full_val[-2]
        val_y_pred = full_val[-1]
        test_y_true = full_test[-2]
        test_y_pred = full_test[-1]
        plot_confusion_matrix(val_y_true, val_y_pred, model_name=type(self.model).__name__, save_path=f"{type(self.model).__name__}_val_confusion.png")
        plot_confusion_matrix(test_y_true, test_y_pred, model_name=type(self.model).__name__, save_path=f"{type(self.model).__name__}_test_confusion.png")
        # t-SNE / PCA 特徵可視化
        if extract_features_after:
            extracted, reduced = perform_feature_extraction_and_reduction(self.model, self.data, self.device, reduction_method=reduction_method, n_components=2, visualize=True, save_path=f"{type(self.model).__name__}_features_{reduction_method}.png")
            # 儲存到 trainer 物件，供報告使用
            self.last_extracted = extracted
            self.last_reduced = reduced
        # 訓練歷史曲線
        plot_training_curves(self.history, save_path=f"{type(self.model).__name__}_training_curves.png", model_name=type(self.model).__name__)
        # 顯示非法類別的度量圖表（Precision / Recall / Macro Recall / AUC）
        try:
            plot_illicit_metric_charts(self.metric_history, model_name=type(self.model).__name__)
        except Exception as e:
            print(f"[警告] 繪製非法類別圖表失敗: {e}")

        # 訓練 PGExplainer 並生成子圖解釋（如果啟用）
        # 注意：這裡需要從全局參數獲取設定，實際使用時會在 main 函數中控制
        use_pg = getattr(self, 'use_pg_explainer', False)
        if use_pg:
            print("\n=== 訓練 PGExplainer 並生成解釋 ===")
            try:
                pg_epochs = getattr(self, 'pg_epochs', 20)
                pg_batch_size = getattr(self, 'pg_batch_size', 64)
                pg_lr = getattr(self, 'pg_lr', 0.003)
                pg_top_k = getattr(self, 'pg_top_k_edges', 50)

                self.train_explainer(epochs=pg_epochs, batch_size=pg_batch_size, lr=pg_lr)
                if self.pg_explainer_trained:
                    self.explain_with_pg_explainer(output_dir=f'{type(self.model).__name__}_pg_explanations', top_k_edges=pg_top_k)
            except Exception as e:
                print(f"[警告] PGExplainer 訓練/解釋失敗: {e}")

        return best_stats

    def train_explainer(self, epochs=20, batch_size=64, lr=0.003):
        """
        訓練 PGExplainer（全域解釋器）來解釋模型如何識別非法交易。

        PGExplainer 是一個全域解釋器，需要使用訓練數據進行訓練，
        然後可以用來解釋整個圖的結構和關鍵路徑。

        Args:
            epochs: PGExplainer 訓練的 epoch 數
            batch_size: 訓練批次大小
            lr: 學習率
        """
        print("=== 開始訓練 PGExplainer（全域解釋器）===")
        print(f"訓練參數: epochs={epochs}, batch_size={batch_size}, lr={lr}")

        try:
            # 創建 PGExplainer
            self.pg_explainer = PGExplainer(
                model=self.model,
                in_channels=self.data.num_node_features,
                device=self.device,
                epochs=epochs,
                lr=lr,
                num_hops=2,  # 考慮 2 跳鄰域
                batch_size=batch_size
            )

            # 準備訓練數據：使用訓練集節點進行訓練
            train_indices = torch.where(self.data.train_mask)[0]

            # 確保有足夠的訓練樣本
            if len(train_indices) == 0:
                print("警告: 沒有訓練樣本可用於訓練 PGExplainer")
                return

            print(f"使用 {len(train_indices)} 個訓練樣本來訓練 PGExplainer")

            # 訓練 PGExplainer
            # PGExplainer 需要整個圖的數據，但只關注訓練節點
            self.pg_explainer.train_explainer(
                x=self.data.x,
                edge_index=self.data.edge_index,
                target=self.data.y,
                index=train_indices
            )

            self.pg_explainer_trained = True
            print("✅ PGExplainer 訓練完成！")

        except Exception as e:
            print(f"❌ PGExplainer 訓練失敗: {e}")
            print("可能的解決方案：")
            print("1. 確保 PyTorch Geometric 版本支持 PGExplainer")
            print("2. 檢查訓練數據是否充足")
            print("3. 嘗試調整訓練參數")

    def explain_with_pg_explainer(self, output_dir='pg_explanations', top_k_edges=50):
        """
        使用訓練好的 PGExplainer 生成子圖解釋，展示模型識別非法交易的關鍵路徑。

        Args:
            output_dir: 輸出目錄
            top_k_edges: 顯示最重要的邊數量
        """
        if not self.pg_explainer_trained or self.pg_explainer is None:
            print("❌ PGExplainer 尚未訓練，請先調用 train_explainer()")
            return

        print("=== 生成 PGExplainer 子圖解釋 ===")
        import matplotlib.pyplot as plt
        import networkx as nx
        from torch_geometric.utils import to_networkx

        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 獲取測試集中預測為非法 (Class 1) 的節點
            self.model.eval()
            with torch.no_grad():
                out = self.model(self.data)
                pred = out.argmax(dim=1)

            test_pred_class_1 = torch.where((self.data.test_mask) & (pred == 1))[0]

            if len(test_pred_class_1) == 0:
                print("警告: 測試集中沒有節點被預測為 Class 1 (非法)")
                return

            print(f"找到 {len(test_pred_class_1)} 個測試集中預測為非法 (Class 1) 的節點")

            # 選擇一個代表性的非法節點進行解釋
            # 選擇預測概率最高的節點
            probs = torch.exp(out).cpu().numpy()
            test_probs_class_1 = probs[test_pred_class_1, 1]
            best_idx = test_pred_class_1[torch.argmax(torch.tensor(test_probs_class_1))]

            print(f"選擇節點 {best_idx.item()} 進行 PGExplainer 解釋（預測概率: {test_probs_class_1.max():.4f}）")

            # 使用 PGExplainer 生成解釋
            explanation = self.pg_explainer(
                x=self.data.x,
                edge_index=self.data.edge_index,
                target=self.data.y,
                index=best_idx
            )

            # 獲取邊重要性
            edge_mask = explanation.edge_mask
            edge_importance = edge_mask.cpu().numpy()

            print(f"生成了 {len(edge_importance)} 條邊的重要性分數")
            print(".4f")

            # 獲取最重要的邊
            top_k = min(top_k_edges, len(edge_importance))
            top_edge_indices = torch.topk(torch.tensor(edge_importance), top_k)[1]
            top_edges = self.data.edge_index[:, top_edge_indices]

            # 獲取相關節點（top edges 的所有節點）
            related_nodes = torch.unique(torch.cat([top_edges[0], top_edges[1]])).cpu().numpy()
            node_mapping = {node: i for i, node in enumerate(related_nodes)}

            print(f"最重要的 {top_k} 條邊涉及 {len(related_nodes)} 個節點")

            # 創建子圖的邊索引
            sub_edge_index = []
            sub_edge_weights = []

            for i, edge_idx in enumerate(top_edge_indices):
                src, dst = self.data.edge_index[:, edge_idx]
                src_mapped = node_mapping.get(src.item())
                dst_mapped = node_mapping.get(dst.item())
                if src_mapped is not None and dst_mapped is not None:
                    sub_edge_index.append([src_mapped, dst_mapped])
                    sub_edge_weights.append(edge_importance[edge_idx])

            if not sub_edge_index:
                print("警告: 無法創建有效的子圖")
                return

            sub_edge_index = torch.tensor(sub_edge_index).T
            sub_edge_weights = torch.tensor(sub_edge_weights)

            # 創建 NetworkX 圖
            G = nx.Graph()
            G.add_nodes_from(range(len(related_nodes)))
            edge_list = sub_edge_index.T.tolist()
            G.add_edges_from(edge_list)

            # 設置節點位置
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

            # 繪製子圖
            plt.figure(figsize=(15, 12))

            # 設置節點顏色
            node_colors = []
            node_sizes = []
            labels = {}

            for i, node_idx in enumerate(related_nodes):
                original_node = node_idx
                labels[i] = str(original_node)

                # 檢查節點是否在測試集中且被預測為非法
                if original_node == best_idx.item():
                    node_colors.append('red')  # 目標節點（被解釋的非法節點）
                    node_sizes.append(400)
                elif self.data.test_mask[original_node] and pred[original_node] == 1:
                    node_colors.append('orange')  # 其他被預測為非法的節點
                    node_sizes.append(250)
                elif self.data.y[original_node] == 1:
                    node_colors.append('darkorange')  # 真實非法的節點
                    node_sizes.append(200)
                else:
                    node_colors.append('lightblue')  # 正常節點
                    node_sizes.append(150)

            # 繪製節點
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                 alpha=0.8, edgecolors='black', linewidths=1)

            # 繪製邊（根據重要性著色）
            if sub_edge_weights.numel() > 0:
                # 將邊權重標準化到 0-1
                normalized_weights = (sub_edge_weights - sub_edge_weights.min()) / (sub_edge_weights.max() - sub_edge_weights.min() + 1e-8)

                # 使用 plasma 色彩圖
                edge_colors = plt.cm.plasma(normalized_weights.numpy())
                edge_widths = 2 + 8 * normalized_weights.numpy()  # 邊寬 2-10

                edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                             width=edge_widths, alpha=0.7,
                                             edge_cmap=plt.cm.plasma)

                # 添加顏色條
                sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                         norm=plt.Normalize(vmin=sub_edge_weights.min().item(),
                                                          vmax=sub_edge_weights.max().item()))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
                cbar.set_label('Edge Importance Score', fontsize=12)

            # 添加節點標籤
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

            # 添加圖例
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                           markersize=15, label='Target Illegal Node'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                           markersize=12, label='Other Predicted Illegal'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
                           markersize=10, label='True Illegal Nodes'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                           markersize=8, label='Normal Nodes')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

            plt.title(f'PGExplainer: Critical Paths for Detecting Illegal Transaction\\n'
                     f'Target Node {best_idx.item()} (Predicted: Illegal)', fontsize=14, pad=20)
            plt.axis('off')
            plt.tight_layout()

            # 保存圖片
            output_path = f'{output_dir}/pg_explainer_subgraph_node_{best_idx.item()}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ PGExplainer 子圖解釋已保存至: {output_path}")
            print(f"圖中顯示了模型識別非法交易時最重要的 {top_k} 條邊")
            print("顏色越深的邊表示重要性越高，紅色節點是正在解釋的目標節點")

        except Exception as e:
            print(f"❌ PGExplainer 解釋生成失敗: {e}")
            import traceback
            traceback.print_exc()

    def apply_self_training(self, threshold=0.9, max_iter=5):
        """
        在訓練完成後應用自訓練半監督學習。
        針對 y == -1 的未知標籤數據生成偽標籤並進行迭代優化。
        """
        print(f"\n=== 開始自訓練 (threshold={threshold}, max_iter={max_iter}) ===")

        # 創建 SelfTraining 實例
        self_training = SelfTraining(self.model, threshold=threshold, max_iter=max_iter)

        # 執行自訓練
        self_training.train_with_pseudo_labels(self.data, self.optimizer, self.criterion, self.device)

        # 重新評估模型性能
        print("\n自訓練完成，重新評估模型性能...")
        full_val, full_test = self.evaluate(detailed=True)

        # 計算最終指標
        val_y_true = full_val[-2]
        val_y_pred = full_val[-1]
        test_y_true = full_test[-2]
        test_y_pred = full_test[-1]

        # 計算各種指標
        val_f1 = f1_score(val_y_true, val_y_pred, average='binary', pos_label=1, zero_division=0)
        val_acc = accuracy_score(val_y_true, val_y_pred)
        test_f1 = f1_score(test_y_true, test_y_pred, average='binary', pos_label=1, zero_division=0)
        test_acc = accuracy_score(test_y_true, test_y_pred)

        try:
            val_precision_illicit = precision_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
            val_recall_illicit = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
            val_macro_recall = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
            val_macro_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        except:
            val_precision_illicit = val_recall_illicit = val_macro_recall = val_macro_f1 = 0.0

        try:
            test_precision_illicit = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_recall_illicit = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_macro_recall = recall_score(test_y_true, test_y_pred, average='macro', zero_division=0)
            test_macro_f1 = f1_score(test_y_true, test_y_pred, average='macro', zero_division=0)
        except:
            test_precision_illicit = test_recall_illicit = test_macro_recall = test_macro_f1 = 0.0

        # 計算 G-Mean
        try:
            val_tpr = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)
            val_tnr = recall_score(val_y_true, val_y_pred, pos_label=0, zero_division=0)
            val_gmean = float((val_tpr * val_tnr) ** 0.5)
        except:
            val_gmean = 0.0

        try:
            test_tpr = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
            test_tnr = recall_score(test_y_true, test_y_pred, pos_label=0, zero_division=0)
            test_gmean = float((test_tpr * test_tnr) ** 0.5)
        except:
            test_gmean = 0.0

        print("自訓練後的最終結果:")
        print(f"驗證集 - F1: {val_f1:.4f}, 準確率: {val_acc:.4f}, Precision: {val_precision_illicit:.4f}, Recall: {val_recall_illicit:.4f}")
        print(f"驗證集 - Macro Recall: {val_macro_recall:.4f}, Macro F1: {val_macro_f1:.4f}, G-Mean: {val_gmean:.4f}")
        print(f"測試集 - F1: {test_f1:.4f}, 準確率: {test_acc:.4f}, Precision: {test_precision_illicit:.4f}, Recall: {test_recall_illicit:.4f}")
        print(f"測試集 - Macro Recall: {test_macro_recall:.4f}, Macro F1: {test_macro_f1:.4f}, G-Mean: {test_gmean:.4f}")

        return {
            'val_f1': val_f1, 'val_acc': val_acc, 'val_precision_illicit': val_precision_illicit,
            'val_recall_illicit': val_recall_illicit, 'val_macro_recall': val_macro_recall,
            'val_macro_f1': val_macro_f1, 'val_gmean': val_gmean,
            'test_f1': test_f1, 'test_acc': test_acc, 'test_precision_illicit': test_precision_illicit,
            'test_recall_illicit': test_recall_illicit, 'test_macro_recall': test_macro_recall,
            'test_macro_f1': test_macro_f1, 'test_gmean': test_gmean
        }


# -----------------------
# Model factory
# -----------------------
MODEL_MAP = {
    'GCN': GCNModel,
    'GAT': GATModel,
    'GraphSAGE': GraphSAGEModel,
    'PCGNN': PC_GNN_Model,
    'GIN': GINModel,
    'SGAT': SGATModel,
    'Hybrid': HybridModel
}


def build_model(name, in_channels, hidden_channels, out_channels, **kwargs):
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {name}")
    # filter kwargs per-model to avoid passing unsupported args (e.g., num_heads to GCN)
    allowed = {
        'GCN': {'num_layers', 'dropout'},
        'GAT': {'num_layers', 'dropout', 'num_heads'},
        'GraphSAGE': {'num_layers', 'dropout', 'aggr'},
        'PCGNN': {'num_layers', 'dropout', 'top_k'},
        'SGAT': {'num_layers', 'dropout', 'num_heads'},
        'Hybrid': {'num_layers', 'dropout', 'num_heads'},
        'GIN': {'num_layers', 'dropout'},
    }
    model_kwargs = {}
    allowed_keys = allowed.get(name, set())
    for k, v in kwargs.items():
        if k in allowed_keys:
            model_kwargs[k] = v
    return MODEL_MAP[name](in_channels, hidden_channels, out_channels, **model_kwargs)


# -----------------------
# CLI entrypoint
# -----------------------
def plot_illicit_metric_charts(metric_history, model_name='Model', output_dir='visualizations'):
    """
    使用 visualization_tools.plot_line_chart 繪製非法類別（label=1）相關指標的曲線圖：
    - Precision (illicit)
    - Recall (illicit)
    - Macro Recall
    - AUC
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(metric_history['val_precision_illicit']) + 1))
    from visualization_tools import plot_line_chart

    charts = [
        ('Precision (非法)', 'val_precision_illicit', 'test_precision_illicit'),
        ('Recall (非法)', 'val_recall_illicit', 'test_recall_illicit'),
        ('Macro Recall', 'val_macro_recall', 'test_macro_recall'),
        ('AUC', 'val_auc', 'test_auc'),
    ]

    for title, val_key, test_key in charts:
        val_vals = metric_history.get(val_key, [])
        test_vals = metric_history.get(test_key, [])
        if len(val_vals) == 0 and len(test_vals) == 0:
            continue
        save_path = os.path.join(output_dir, f"{model_name.lower()}_{val_key}.png")
        # 使用 plot_line_chart，傳入 epochs 與兩條線（驗證、測試）
        try:
            plot_line_chart(epochs, [val_vals, test_vals], labels=['驗證', '測試'], title=f"{model_name} - {title}", xlabel='Epoch', ylabel=title, save_path=save_path)
        except Exception:
            # fallback: 簡單手動畫圖（避免完全失敗）
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            if len(val_vals) > 0:
                plt.plot(epochs, val_vals, label='驗證', marker='o')
            if len(test_vals) > 0:
                plt.plot(epochs, test_vals, label='測試', marker='o')
            plt.xlabel('Epoch'); plt.ylabel(title); plt.title(f"{model_name} - {title}"); plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unified GNN training framework')
    parser.add_argument('--model', type=str, default='GCN', choices=list(MODEL_MAP.keys()))
    parser.add_argument('--ensemble_mode', type=str, default=None, choices=['gcn_gat_graphsage', 'gcn_gat_gin', 'gin_gat_graphsage'], help='啟用集成訓練模式：訓練多個模型並集成預測')
    parser.add_argument('--dataset_dir', type=str, default='../Dataset')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--loss_type', type=str, default=None, choices=['nll', 'focal'], help='若為 None，將根據模型自動選擇（Advanced -> focal, else -> nll）')
    parser.add_argument('--reduction_method', type=str, default='pca', choices=['pca', 'tsne'])
    parser.add_argument('--no_extract_features', action='store_true')
    # Optuna related args
    parser.add_argument('--optuna', action='store_true', help='啟用 Optuna 自動化超參數搜尋')
    parser.add_argument('--optuna_mode', type=str, default='trainer', choices=['trainer', 'original'], help='Optuna 優化模式: "trainer" 使用簡化的參數優化, "original" 使用原始的完整參數搜索')
    parser.add_argument('--optuna_trials', type=int, default=20, help='Optuna 搜尋的試驗次數')
    parser.add_argument('--optuna_epochs', type=int, default=20, help='Optuna 每次試驗的訓練 epoch（推薦較小以加速搜尋）')
    parser.add_argument('--use_graph_smote', action='store_true', help='訓練前使用 Graph-SMOTE 增強少數類')
    parser.add_argument('--smote_ratio', type=float, default=1.0, help='Graph-SMOTE 新增樣本比例（相對於少數類）')
    parser.add_argument('--smote_k', type=int, default=5, help='Graph-SMOTE 中選擇近鄰數 k（簡化為隨機選擇）')
    parser.add_argument('--label_balance', action='store_true', help='在每個 epoch 的訓練 step 中使用 label-balanced sampler')
    parser.add_argument('--batch_size_balance', type=int, default=128, help='label-balanced sampler 的 batch size（每次訓練使用子集 loss）')
    # GAN args
    parser.add_argument('--use_gan', action='store_true', help='啟用 GAN 增強少數類（WGAN-GP）')
    parser.add_argument('--gan_steps', type=int, default=5, help='每 epoch GAN 的訓練步數（D/G 交替）')
    parser.add_argument('--gan_batch_size', type=int, default=64, help='GAN 訓練批次大小')
    parser.add_argument('--gan_latent_dim', type=int, default=32, help='GAN 噪聲維度')
    parser.add_argument('--gan_lr', type=float, default=1e-4, help='GAN 優化器學習率')
    parser.add_argument('--gan_gp_lambda', type=float, default=10.0, help='WGAN-GP gradient penalty lambda')
    parser.add_argument('--gan_gen_ratio', type=float, default=1.0, help='每 epoch 生成樣本數相對於少數類的比例')
    # SelfTraining args
    parser.add_argument('--use_self_training', action='store_true', help='在初始訓練完成後啟用自訓練半監督學習')
    parser.add_argument('--self_training_threshold', type=float, default=0.9, help='自訓練的信心閾值（生成偽標籤的概率閾值）')
    parser.add_argument('--self_training_max_iter', type=int, default=5, help='自訓練的最大迭代次數')
    # Pre-training GAN args
    parser.add_argument('--use_pretrain_gan', action='store_true', help='在 GNN 訓練前先預訓練 GAN 生成額外的非法樣本')
    parser.add_argument('--pretrain_gan_epochs', type=int, default=100, help='預訓練 GAN 的 epoch 數')
    parser.add_argument('--pretrain_gan_batch_size', type=int, default=64, help='預訓練 GAN 的批次大小')
    parser.add_argument('--pretrain_gan_lr', type=float, default=1e-4, help='預訓練 GAN 的學習率')
    parser.add_argument('--pretrain_gan_latent_dim', type=int, default=32, help='預訓練 GAN 的噪聲維度')
    parser.add_argument('--pretrain_gan_augment_ratio', type=float, default=1.0, help='生成的合成樣本數相對於原始少數類的比例')
    # Model Explanation args
    parser.add_argument('--explain_model', action='store_true', help='訓練後使用 GNNExplainer 解釋模型，針對預測為非法 (Class 1) 的測試節點')
    parser.add_argument('--explain_samples', type=int, default=5, help='要解釋的樣本數量（從預測為非法類的測試節點中隨機選擇）')
    parser.add_argument('--explanation_dir', type=str, default='explanations', help='解釋結果輸出目錄')
    # PGExplainer args
    parser.add_argument('--use_pg_explainer', action='store_true', help='訓練後使用 PGExplainer（全域解釋器）生成子圖解釋')
    parser.add_argument('--pg_epochs', type=int, default=20, help='PGExplainer 訓練的 epoch 數')
    parser.add_argument('--pg_batch_size', type=int, default=64, help='PGExplainer 訓練批次大小')
    parser.add_argument('--pg_lr', type=float, default=0.003, help='PGExplainer 學習率')
    parser.add_argument('--pg_top_k_edges', type=int, default=50, help='PGExplainer 解釋中顯示的最重要邊數量')
    args = parser.parse_args()

    data = load_elliptic_data(args.dataset_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 預訓練 GAN 生成額外的非法樣本
    if args.use_pretrain_gan:
        print("=== 開始預訓練 GAN 生成額外的非法樣本 ===")
        data = pretrain_gan_and_augment(
            data, device, args.pretrain_gan_epochs, args.pretrain_gan_batch_size,
            args.pretrain_gan_lr, args.pretrain_gan_latent_dim, args.pretrain_gan_augment_ratio
        )

    in_ch = data.num_node_features
    out_ch = 2
    model_kwargs = {'num_layers': args.num_layers, 'dropout': args.dropout, 'num_heads': args.num_heads}
    model = build_model(args.model, in_ch, args.hidden_channels, out_ch, **model_kwargs)

    # 損失選擇
    if args.loss_type:
        loss_type = args.loss_type
    else:
        loss_type = 'focal' if args.model in ['SGAT', 'Hybrid'] else 'nll'
    if loss_type == 'nll':
        criterion = nn.NLLLoss()
    else:
        criterion = FocalLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = Trainer(model, data, device, optimizer, criterion)
    # apply label-balanced sampler settings if requested
    if args.label_balance:
        trainer.label_balance = True
        trainer.batch_size = args.batch_size_balance
    else:
        trainer.label_balance = False
        trainer.batch_size = None
    # PGExplainer settings
    if args.use_pg_explainer:
        trainer.use_pg_explainer = True
        trainer.pg_epochs = args.pg_epochs
        trainer.pg_batch_size = args.pg_batch_size
        trainer.pg_lr = args.pg_lr
        trainer.pg_top_k_edges = args.pg_top_k_edges
    else:
        trainer.use_pg_explainer = False
    # GAN setup
    if args.use_gan:
        feat_dim = data.num_node_features
        gen = GeneratorMLP(args.gan_latent_dim, feat_dim, hidden_dim=max(128, feat_dim))
        dis = DiscriminatorMLP(feat_dim, hidden_dim=max(128, feat_dim))
        g_opt = torch.optim.Adam(gen.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(dis.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))
        trainer.gan_enabled = True
        trainer.gan_generator = gen
        trainer.gan_discriminator = dis
        trainer.gan_g_optimizer = g_opt
        trainer.gan_d_optimizer = d_opt
        trainer.gan_steps = args.gan_steps
        trainer.gan_batch_size = args.gan_batch_size
        trainer.gan_latent_dim = args.gan_latent_dim
        trainer.gan_gp_lambda = args.gan_gp_lambda
        trainer.gan_gen_ratio = args.gan_gen_ratio
    else:
        trainer.gan_enabled = False
    # 如果啟用 Optuna，執行超參數搜尋
    if args.optuna:
        def run_trainer_optuna_search(data, args, device):
            """
            針對 Trainer 流程的 Optuna 超參數優化。
            優化 lr, hidden_channels, dropout，並根據模型類型決定是否優化 num_heads。
            """
            def trainer_objective(trial):
                # 基本參數搜索空間
                lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
                dropout = trial.suggest_uniform('dropout', 0.2, 0.5)

                # 根據模型類型決定是否優化 num_heads
                if args.model in ['GAT', 'Hybrid']:
                    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
                    model_kwargs = {'num_layers': args.num_layers, 'dropout': dropout, 'num_heads': num_heads}
                else:
                    # GCN, GraphSAGE, GIN 等模型不需要 num_heads
                    model_kwargs = {'num_layers': args.num_layers, 'dropout': dropout}

                # 建立模型與訓練器
                local_data = copy.deepcopy(data)

                # 可選的 Graph-SMOTE
                if args.use_graph_smote:
                    try:
                        local_data = graph_smote(local_data, ratio=args.smote_ratio, k=args.smote_k, device=device)
                    except Exception as e:
                        print(f"[警告] Optuna 試驗中 Graph-SMOTE 失敗: {e}")

                # 建立模型
                model = build_model(args.model, local_data.num_node_features, hidden_channels, 2, **model_kwargs).to(device)

                # 損失函數選擇
                loss_type = args.loss_type if args.loss_type else ('focal' if args.model in ['SGAT', 'Hybrid'] else 'nll')
                criterion = nn.NLLLoss() if loss_type == 'nll' else FocalLoss()

                # 優化器
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

                # 創建 Trainer
                trainer = Trainer(model, local_data, device, optimizer, criterion)

                # 應用 label-balanced sampler 如果啟用
                if args.label_balance:
                    trainer.label_balance = True
                    trainer.batch_size = args.batch_size_balance

                # 訓練模型
                trainer.fit(epochs=args.optuna_epochs, extract_features_after=False, reduction_method=args.reduction_method)

                # 評估驗證集性能
                val_res, _ = trainer.evaluate(detailed=False)
                val_f1 = val_res[1]  # F1 分數

                # 返回驗證 F1 分數作為優化目標
                return float(val_f1)

            # 創建 Optuna study
            study = optuna.create_study(direction='maximize')

            print(f"開始 Optuna 超參數優化 ({args.optuna_trials} 試驗)...")
            print(f"優化參數: lr, hidden_channels, dropout" + (", num_heads" if args.model in ['GAT', 'Hybrid'] else ""))

            # 執行優化
            study.optimize(trainer_objective, n_trials=args.optuna_trials)

            # 輸出最佳結果
            print(f"\nOptuna 優化完成!")
            print(f"最佳驗證 F1: {study.best_trial.value:.4f}")
            print("最佳參數:")
            for param_name, param_value in study.best_trial.params.items():
                print(f"  {param_name}: {param_value}")

            return study

        def run_optuna_search(data, args, device):
            def objective(trial):
                # 搜索空間
                hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
                dropout = trial.suggest_uniform('dropout', 0.0, 0.7)
                num_layers = trial.suggest_int('num_layers', 2, 4)
                num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
                # augmentation params
                edge_drop_rate = trial.suggest_uniform('edge_drop_rate', 0.0, 0.3)
                feature_noise_level = trial.suggest_uniform('feature_noise_level', 0.0, 0.05)

                # 建立模型與訓練器（使用 data 的複本以避免交叉污染）
                local_data = copy.deepcopy(data)
                # optional Graph-SMOTE in optuna trials
                if args.use_graph_smote:
                    try:
                        local_data = graph_smote(local_data, ratio=args.smote_ratio, k=args.smote_k, device=device)
                    except Exception as e:
                        print(f"[警告] Optuna 試驗中 Graph-SMOTE 失敗: {e}")
                model_kwargs = {'num_layers': num_layers, 'dropout': dropout, 'num_heads': num_heads}
                model = build_model(args.model, local_data.num_node_features, hidden_channels, 2, **model_kwargs).to(device)
                loss_type = args.loss_type if args.loss_type else ('focal' if args.model in ['SGAT', 'Hybrid'] else 'nll')
                criterion = nn.NLLLoss() if loss_type == 'nll' else FocalLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

                # 將 augmentation 參數注入到 trainer 的 data augmentation（透過簡單替換 simple_graph_augmentation 內的預設）
                # 這裡我們暫時直接在 trainer.fit 之前用 aug data 訓練（trainer.fit 本身也會再做 augmentation）
                trainer = Trainer(model, local_data, device, optimizer, criterion)
                # 在試驗中使用較少的 epoch 以加速
                trainer.fit(epochs=args.optuna_epochs, extract_features_after=False, reduction_method=args.reduction_method)
                # 評估並以驗證 F1 作為目標
                val_res, _ = trainer.evaluate(detailed=False)
                val_loss = val_res[0]
                val_f1 = val_res[1]
                # 告訴 Optuna 這次試驗的結果
                return float(val_f1)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=args.optuna_trials)
            print("Optuna 最佳 trial:", study.best_trial.params)
            return study

        if args.optuna_mode == 'trainer':
            study = run_trainer_optuna_search(data, args, device)
        else:  # original mode
            study = run_optuna_search(data, args, device)
        best_params = study.best_trial.params
        # 用最佳參數建立最終模型並完整訓練
        print("使用最佳參數重新訓練完整模型...")
        best_hidden = best_params.get('hidden_channels', args.hidden_channels)
        best_lr = best_params.get('learning_rate', args.learning_rate)
        best_dropout = best_params.get('dropout', args.dropout)
        best_num_layers = best_params.get('num_layers', args.num_layers)
        best_num_heads = best_params.get('num_heads', args.num_heads)
        model_kwargs = {'num_layers': best_num_layers, 'dropout': best_dropout, 'num_heads': best_num_heads}
        model = build_model(args.model, in_ch, best_hidden, out_ch, **model_kwargs)
        loss_type = args.loss_type if args.loss_type else ('focal' if args.model in ['SGAT', 'Hybrid'] else 'nll')
        criterion = nn.NLLLoss() if loss_type == 'nll' else FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=args.weight_decay)
        # 在最終完整訓練前可選擇應用 Graph-SMOTE
        if args.use_graph_smote:
            try:
                data = graph_smote(data, ratio=args.smote_ratio, k=args.smote_k, device=device)
                print(f"[Info] 已對資料套用 Graph-SMOTE，新增樣本比例: {args.smote_ratio}")
            except Exception as e:
                print(f"[警告] Graph-SMOTE 應用失敗，將使用原始資料訓練: {e}")
        trainer = Trainer(model, data, device, optimizer, criterion)
        best_stats = trainer.fit(epochs=args.epochs, extract_features_after=(not args.no_extract_features), reduction_method=args.reduction_method)
        print("最佳統計：", best_stats)

        # 在初始訓練完成後應用自訓練
        if args.use_self_training:
            self_training_stats = trainer.apply_self_training(
                threshold=args.self_training_threshold,
                max_iter=args.self_training_max_iter
            )
            print("自訓練統計：", self_training_stats)
    elif args.ensemble_mode:
        # 集成訓練模式
        def train_ensemble_models(model_names, ensemble_class):
            """訓練多個模型並返回集成模型"""
            trained_models = []
            histories = []

            for model_name in model_names:
                print(f"\n訓練模型: {model_name}")
                model_kwargs = {'num_layers': args.num_layers, 'dropout': args.dropout, 'num_heads': args.num_heads}
                model = build_model(model_name, in_ch, args.hidden_channels, out_ch, **model_kwargs)

                # 損失選擇
                loss_type = args.loss_type if args.loss_type else ('focal' if model_name in ['SGAT', 'Hybrid'] else 'nll')
                criterion = nn.NLLLoss() if loss_type == 'nll' else FocalLoss()

                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                trainer = Trainer(model, data, device, optimizer, criterion)

                # 應用 label-balanced sampler settings
                if args.label_balance:
                    trainer.label_balance = True
                    trainer.batch_size = args.batch_size_balance

                # 應用 Graph-SMOTE
                if args.use_graph_smote:
                    try:
                        train_data = graph_smote(data, ratio=args.smote_ratio, k=args.smote_k, device=device)
                        trainer.data = train_data
                        print(f"[Info] 已對 {model_name} 套用 Graph-SMOTE")
                    except Exception as e:
                        print(f"[警告] {model_name} Graph-SMOTE 應用失敗: {e}")

                # 訓練模型
                trainer.fit(epochs=args.epochs, extract_features_after=False, reduction_method=args.reduction_method)

                # 對每個模型應用自訓練
                if args.use_self_training:
                    print(f"對模型 {model_name} 應用自訓練...")
                    self_training_stats = trainer.apply_self_training(
                        threshold=args.self_training_threshold,
                        max_iter=args.self_training_max_iter
                    )
                    print(f"模型 {model_name} 自訓練完成")

                trained_models.append(model)
                histories.append(trainer.history)

            # 創建集成模型
            ensemble_model = ensemble_class(trained_models, voting='soft')
            return ensemble_model, histories

        if args.ensemble_mode == 'gcn_gat_graphsage':
            print("\n=== 訓練集成模型：GCN + GAT + GraphSAGE ===")
            model_names = ['GCN', 'GAT', 'GraphSAGE']
            ensemble, histories = train_ensemble_models(model_names, EnsembleModel)

            # 使用集成模型進行預測
            ensemble_pred = ensemble.predict(data, device)
            y_true = data.y[data.test_mask].cpu().numpy()

            print("\n集成模型 (GCN + GAT + GraphSAGE) 分類報告:")
            print(classification_report(y_true, ensemble_pred, target_names=['合法', '非法'], zero_division=0))

            # 計算其他指標
            f1_ensemble = f1_score(y_true, ensemble_pred, average='binary', pos_label=1, zero_division=0)
            acc_ensemble = accuracy_score(y_true, ensemble_pred)
            prec_ensemble = precision_score(y_true, ensemble_pred, pos_label=1, zero_division=0)
            recall_ensemble = recall_score(y_true, ensemble_pred, pos_label=1, zero_division=0)
            macro_recall = recall_score(y_true, ensemble_pred, average='macro', zero_division=0)
            macro_f1 = f1_score(y_true, ensemble_pred, average='macro', zero_division=0)

            try:
                auc_ensemble = roc_auc_score(y_true, ensemble_pred)  # 這裡用預測結果近似
                macro_auc = auc_ensemble  # 對於二分類，macro AUC 等於普通 AUC
            except:
                auc_ensemble = float('nan')
                macro_auc = float('nan')

            print(f"集成模型測試 F1: {f1_ensemble:.4f}, 準確率: {acc_ensemble:.4f}, Precision: {prec_ensemble:.4f}, Recall: {recall_ensemble:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, AUC: {auc_ensemble:.4f}")

        elif args.ensemble_mode == 'gcn_gat_gin':
            print("\n=== 訓練集成模型：GCN + GAT + GIN ===")
            model_names = ['GCN', 'GAT', 'GIN']
            ensemble, histories = train_ensemble_models(model_names, EnsembleModel)

            # 使用集成模型進行預測
            ensemble_pred = ensemble.predict(data, device)
            y_true = data.y[data.test_mask].cpu().numpy()

            print("\n集成模型 (GCN + GAT + GIN) 分類報告:")
            print(classification_report(y_true, ensemble_pred, target_names=['合法', '非法'], zero_division=0))

            # 計算其他指標
            f1_ensemble = f1_score(y_true, ensemble_pred, average='binary', pos_label=1, zero_division=0)
            acc_ensemble = accuracy_score(y_true, ensemble_pred)
            prec_ensemble = precision_score(y_true, ensemble_pred, pos_label=1, zero_division=0)
            recall_ensemble = recall_score(y_true, ensemble_pred, pos_label=1, zero_division=0)
            macro_recall = recall_score(y_true, ensemble_pred, average='macro', zero_division=0)
            macro_f1 = f1_score(y_true, ensemble_pred, average='macro', zero_division=0)

            try:
                auc_ensemble = roc_auc_score(y_true, ensemble_pred)  # 這裡用預測結果近似
                macro_auc = auc_ensemble  # 對於二分類，macro AUC 等於普通 AUC
            except:
                auc_ensemble = float('nan')
                macro_auc = float('nan')

            print(f"集成模型測試 F1: {f1_ensemble:.4f}, 準確率: {acc_ensemble:.4f}, Precision: {prec_ensemble:.4f}, Recall: {recall_ensemble:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, AUC: {auc_ensemble:.4f}")

        elif args.ensemble_mode == 'gin_gat_graphsage':
            print("\n=== 訓練集成模型：GIN + GAT + GraphSAGE ===")
            model_names = ['GIN', 'GAT', 'GraphSAGE']
            ensemble, histories = train_ensemble_models(model_names, EnsembleModel)

            # 使用集成模型進行預測
            ensemble_pred = ensemble.predict(data, device)
            y_true = data.y[data.test_mask].cpu().numpy()

            print("\n集成模型 (GIN + GAT + GraphSAGE) 分類報告:")
            print(classification_report(y_true, ensemble_pred, target_names=['合法', '非法'], zero_division=0))

            # 計算其他指標
            f1_ensemble = f1_score(y_true, ensemble_pred, average='binary', pos_label=1, zero_division=0)
            acc_ensemble = accuracy_score(y_true, ensemble_pred)
            prec_ensemble = precision_score(y_true, ensemble_pred, pos_label=1, zero_division=0)
            recall_ensemble = recall_score(y_true, ensemble_pred, pos_label=1, zero_division=0)
            macro_recall = recall_score(y_true, ensemble_pred, average='macro', zero_division=0)
            macro_f1 = f1_score(y_true, ensemble_pred, average='macro', zero_division=0)

            try:
                auc_ensemble = roc_auc_score(y_true, ensemble_pred)  # 這裡用預測結果近似
                macro_auc = auc_ensemble  # 對於二分類，macro AUC 等於普通 AUC
            except:
                auc_ensemble = float('nan')
                macro_auc = float('nan')

            print(f"集成模型測試 F1: {f1_ensemble:.4f}, 準確率: {acc_ensemble:.4f}, Precision: {prec_ensemble:.4f}, Recall: {recall_ensemble:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, AUC: {auc_ensemble:.4f}")

    else:
        best_stats = trainer.fit(epochs=args.epochs, extract_features_after=(not args.no_extract_features), reduction_method=args.reduction_method)
        print("最佳統計：", best_stats)

        # 在初始訓練完成後應用自訓練
        if args.use_self_training:
            self_training_stats = trainer.apply_self_training(
                threshold=args.self_training_threshold,
                max_iter=args.self_training_max_iter
            )
            print("自訓練統計：", self_training_stats)

    # 訓練結束後自動生成完整報告（模型、history、embeds）
    try:
        from visualization_tools import run_full_report
        embeds = getattr(trainer, 'last_reduced', None)
        run_full_report(model, data, trainer.history, embeds=embeds, output_dir=f'visualizations/{type(model).__name__}_full_report', reduction_method=args.reduction_method)
    except Exception as e:
        print(f"[警告] 生成完整報告失敗: {e}")

    # 模型解釋：針對預測為非法 (Class 1) 的測試節點進行解釋
    if args.explain_model:
        try:
            print("=== 開始模型解釋 ===")
            explain_model(model, data, device, num_samples=args.explain_samples, output_dir=args.explanation_dir)
        except Exception as e:
            print(f"[警告] 模型解釋失敗: {e}")
            print("可能的解決方案：")
            print("1. 確保安裝了 networkx 和 seaborn: pip install networkx seaborn")
            print("2. 檢查 PyTorch Geometric 版本是否支持 GNNExplainer")


if __name__ == '__main__':
    main()


