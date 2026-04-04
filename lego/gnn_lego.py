"""
The Hong Kong Polytechnic University
Student ID: 24027277d
Name: Yuen Tsz Ki

GNN ZOO
"""

import torch
import torch.nn as nn
from torch_geometric.nn.encoding import TemporalEncoding
from torch_geometric.nn import MixHopConv, APPNP, GATConv
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
        K = best_params.get('K', "3")
        alpha = best_params.get('alpha', "0.2")


        self.dropout = best_params.get('dropout', 0.5)
        
        # 1. 定義可學習的層 (Layer Definitions)
        self.comm_emb = nn.Embedding(num_communities, 16)
        self.temp_enc = TemporalEncoding(out_channels=16)
        
        # 假設 PPR 和 Degree 已經算出並作為節點特徵
        # 總輸入維度 = 原始(in) + Community(16) + PPR(ppr_dim) + Degree(degree_dim)
        total_in_channels = in_channels + 16 + 16
        
        # ----- Branch 1: MixHop -----
        self.mixhop = MixHopConv(total_in_channels, hidden_channels, powers=[0, 1, 2])
        self.mixhop_proj = nn.Linear(hidden_channels * 3, hidden_channels)

        # ----- Branch 2: GAT -----
        self.gat = GATConv(total_in_channels, hidden_channels, heads=1, concat=True)

        # ----- Branch 3: APPNP-style -----
        self.appnp_mlp = nn.Linear(total_in_channels, hidden_channels)
        self.appnp_prop = APPNP(K=K, alpha=alpha)

        # Gating network: 對每個節點產生 3 個分支權重
        self.gate = nn.Linear(hidden_channels * 3, 3)

        # 最後線性分類頭
        self.classifier = nn.Linear(hidden_channels, out_channels)

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
        
        h_mix = self.mixhop(x, edge_index)
        h_mix = F.relu(self.mixhop_proj(h_mix))

        h_gat = F.relu(self.gat(x, edge_index))

        h_app = F.relu(self.appnp_mlp(x))
        h_app = self.appnp_prop(h_app, edge_index)

        h_mix = F.dropout(h_mix, p=self.dropout, training=self.training)
        h_gat = F.dropout(h_gat, p=self.dropout, training=self.training)
        h_app = F.dropout(h_app, p=self.dropout, training=self.training)

        # concat for gating
        h_cat = torch.cat([h_mix, h_gat, h_app], dim=-1)   # [N, 3H]
        w = torch.softmax(self.gate(h_cat), dim=-1)         # [N, 3]

        # node-wise weighted fusion
        h = (
            w[:, 0:1] * h_mix +
            w[:, 1:2] * h_gat +
            w[:, 2:3] * h_app
        )  # [N, H]

        # linear classifier head
        out = self.classifier(h)  # [N, out_channels]
        return out
