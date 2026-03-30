"""
The Hong Kong Polytechnic University
Student ID: 24027277d
Name: Yuen Tsz Ki

GNN ZOO
"""

import torch
import torch.nn as nn
from torch_geometric.nn.encoding import TemporalEncoding
from torch_geometric.nn import MixHopConv, APPNP
import torch.nn.functional as F


class Lego_GNN(torch.nn.Module):
    def __init__(self, num_communities, best_params):
        super().__init__()

        # Official PyG Mixhop
        in_channels = best_params.get('in_channels', None)
        hidden_channels = best_params.get('hidden_channels', 64)
        out_channels = 2
        powers = best_params.get('powers', "[0, 1, 2]")
        powers = eval(powers)
        current_dim = hidden_channels * len(powers)

        self.dropout = best_params.get('dropout', 0.5)
        
        # 1. 定義可學習的層 (Layer Definitions)
        self.comm_emb = nn.Embedding(num_communities, 16)
        self.temp_enc = TemporalEncoding(out_channels=16)
        
        # 假設 PPR 和 Degree 已經算出並作為節點特徵
        # 總輸入維度 = 原始(in) + Community(16) + PPR(ppr_dim) + Degree(degree_dim)
        total_in_channels = in_channels + 16 + 16
        
        # MixHopConv 使用 powers=[0, 1, 2] 代表同時考慮 0, 1, 2 階鄰居
        self.conv1 = MixHopConv(total_in_channels, hidden_channels, powers=powers)
        
        # MixHop 的輸出維度是 hidden_channels * len(powers)
        current_dim = hidden_channels * len(powers)
        
        self.conv2 = MixHopConv(current_dim, hidden_channels, powers=powers)
        final_dim = hidden_channels * len(powers)
        
        self.bn1 = nn.BatchNorm1d(current_dim)
        self.bn2 = nn.BatchNorm1d(final_dim)
        
        self.fc = nn.Linear(final_dim, out_channels) # 二分類
        self.appnp = APPNP(K=10, alpha=0.875)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        comm_id = data.comm_id.long() # 確保是整數索引
        timesteps= data.timesteps.long()
        
        # 獲取預處理好的 PPR 和 Degree (假設已存在 data 中)
        # ppr_score = data.ppr_score # [num_nodes, 1]
        # deg_feat = data.degree # [num_nodes, 1]
        time_feat = self.temp_enc(timesteps.float())
        # 1. 取得 Community Embedding [num_nodes, 16]
        comm_feat = self.comm_emb(comm_id)
        
        # 2. 拼接所有節點層級的特徵 (Node-level Fusion)
        # 注意：所有拼接的 Tensor 必須在 dim=0 上長度一致 (num_nodes)
        x = torch.cat([x, comm_feat, time_feat], dim=-1)
        
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. 第一層卷積與後處理
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 4. 第二層卷積
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 5. 分類輸出
        return self.fc(x)