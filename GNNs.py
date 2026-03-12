## GNN model ZOO

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE, Sequential
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP, APPNP, ChebConv, GCN2Conv, MixHopConv

from torch.nn import ReLU, Dropout, BatchNorm1d, Softmax, LogSoftmax, Linear

# =================================================================================
# basic GNNs - done
# =================================================================================
# model = GCN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm').to(device)
# model = GAT(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', heads=heads).to(device)
# model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', aggr='mean').to(device)    
# model = GIN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm').to(device)

def GCNModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    
    return GCN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm')

def GATModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    heads = best_params.get('heads', 8)
    return GAT(in_channels, hidden_channels, num_layers, out_channels, dropout, heads=heads)

def GraphSAGEModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    return GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, dropout)

def GINModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    return GIN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm')

# =================================================================================
# APPNP compare 
# --------- done ---------
# APPNP 2 layer version 的表現很差，可能是因為 APPNP 的特性使得過多的層數反而會導致over-smoothing問題，從而降低模型的表現。
# APPNP 2 layer version 的 recall, F1全部是0
# =================================================================================
# def APPNPModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=10, alpha=0.1):
#     return Sequential('x, edge_index', [
#         ##    input layer, input layer ->   hidden layer
#         (MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout), 'x -> x'),
       
#         ##    hidden layer,    hidden layer -> output layer
#         (MLP([hidden_channels, out_channels], norm=None), 'x -> x'),

#         # output layer
#         (APPNP(K=K, alpha=alpha, dropout=dropout), 'x, edge_index -> x'),
#         (LogSoftmax(dim=1), 'x -> x')
#     ])

def APPNPModel_1Layer(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    K = best_params.get('K', 10)
    alpha = best_params.get('alpha', 0.1)

    return Sequential('x, edge_index', [
        ##    input layer, input layer ->   hidden layer
        (MLP([in_channels, hidden_channels, out_channels], dropout=dropout), 'x -> x'),

        # output layer
        (APPNP(K=K, alpha=alpha, dropout=dropout), 'x, edge_index -> x'),
        (LogSoftmax(dim=1), 'x -> x')
    ])

# =================================================================================
# ChebNet pure v.s. linear 
# AI建議使用 linear version, 
# 結果亦顯示 linear 表現比較好
# --------- done ---------
# 留 linear version
# =================================================================================
# def PureChebNetModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
#     return Sequential('x, edge_index', [
#         ## input layer
#         (ChebConv(in_channels, hidden_channels, K=K), 'x, edge_index -> x'),
#         (BatchNorm1d(hidden_channels), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(dropout), 'x -> x'),

#         ## hidden layer
#         (ChebConv(hidden_channels, hidden_channels, K=K), 'x, edge_index -> x'),
#         (BatchNorm1d(hidden_channels), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(dropout), 'x -> x'),

#         ## output layer
#         (ChebConv(hidden_channels, out_channels, K=K), 'x, edge_index -> x'),
#         (LogSoftmax(dim=1), 'x -> x'),
#     ])

def LinearChebNetModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    K = best_params.get('K', 3)

    return Sequential('x, edge_index', [
        ## input layer
        (ChebConv(in_channels, hidden_channels, K=K), 'x, edge_index -> x'),
        (BatchNorm1d(hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(dropout), 'x -> x'),

        ## hidden layer
        (ChebConv(hidden_channels, hidden_channels, K=K), 'x, edge_index -> x'),
        (BatchNorm1d(hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(dropout), 'x -> x'),

        ## output layer (Linear(64, out_channels), 'x -> x'),
        (Linear(hidden_channels, out_channels), 'x -> x'),
        (LogSoftmax(dim=1), 'x -> x'),
    ])

# =================================================================================
# GCN - hidden layer 不用 Dropout v.s.  hidden layer保留Dropout
# AI建議hidden layer不用Dropout 
# 結果亦顯示 hidden layer 不用Dropout 表現比較好
# AI 建議可再加normalization, 殘差連接 (Residual Connection)
# --------- done ---------
# 1 dropout version 的表現比2 dropout version好 hidden layer不同dropout的表現更佳
# =================================================================================
# def MixHopGCNModel(in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
#     ## Loss 曲線平滑且緩慢下降：這是好事，代表正規化在發揮作用。
#     ## Training Loss 降不下來：這代表正規化過頭了。
#     ## 全表現比GCN MixHop 更好

#     mix_out = hidden_channels * len(powers)

#     return Sequential('x, edge_index', [
#         (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
#         (BatchNorm1d(mix_out), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(dropout), 'x -> x'),

#         (GCNConv(mix_out, hidden_channels), 'x, edge_index -> x'),
#         (BatchNorm1d(hidden_channels), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(dropout), 'x -> x'),

#         (Linear(hidden_channels, out_channels), 'x -> x'),
#         (LogSoftmax(dim=1), 'x -> x'),
#     ])


def MixHopGCNModel_1dropout(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    powers = best_params.get('powers', [0, 1, 2])

    ## Loss 曲線平滑且緩慢下降：這是好事，代表正規化在發揮作用。
    ## Training Loss 降不下來：這代表正規化過頭了。
    # mixhop_dim_multiplier = len(powers) 
    mix_out = hidden_channels * len(powers)

    return Sequential('x, edge_index', [
        (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm1d(mix_out), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(dropout), 'x -> x'),

        (GCNConv(mix_out, hidden_channels), 'x, edge_index -> x'),
        (BatchNorm1d(hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),

        (Linear(hidden_channels, out_channels), 'x -> x'),
        (LogSoftmax(dim=1), 'x -> x'),
    ])


# =================================================================================
# GAT - done
# AI: 1.如果發現模型在稀疏數據集上不收斂，可以考慮單獨設置一個較小的 attn_dropout（例如 0.2），而保留結構層級的 dropout 為 0.5。
#     2. 追求極致的泛化，可在 Linear 之前加入殘差
# =================================================================================

def MixHopGATModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    powers = best_params.get('powers', [0, 1, 2])
    heads = best_params.get('heads', 8)
    mix_out = hidden_channels * len(powers)
    
    return Sequential('x, edge_index', [
        (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm1d(mix_out), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (GATConv(mix_out, hidden_channels, heads, dropout=dropout), 'x, edge_index -> x'),
        (BatchNorm1d(hidden_channels * heads), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (Linear(hidden_channels * heads, out_channels), 'x -> x'),
        (LogSoftmax(dim=1), 'x -> x'),
    ])

# =================================================================================
# GraphSAGE normal vs. bottleneck layer(特徵蒸餾)
# AI: SAGEConv 中加入 normalize=True
# 特徵蒸餾: MixHopGAT, MixHopGCN 可加其他不建議
#          (Linear(mix_out, hidden_channels), 'x -> x'),
#          (ReLU(inplace=True), 'x -> x'),
# --------- done ---------
# 沒有bottleneck layer的版本表現極差，precision, recall, F1全部是0
# =================================================================================

# def MixHopGraphSAGEModel(in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
#     mix_out = hidden_channels * len(powers)

#     return Sequential('x, edge_index', [
#         (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
#         (BatchNorm1d(mix_out), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(p=dropout), 'x -> x'),

#         ## aggr lstm, max
#         (SAGEConv(mix_out, hidden_channels), 'x, edge_index -> x'),  
#         (BatchNorm1d(hidden_channels), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(p=dropout), 'x -> x'),

#         (Linear(hidden_channels, out_channels), 'x -> x'),
#         (LogSoftmax(dim=1), 'x -> x'),
#     ])

def MixHopGraphSAGEModel_Bottleneck(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    powers = best_params.get('powers', [0, 1, 2])
    mix_out = hidden_channels * len(powers)

    return Sequential('x, edge_index', [
        (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm1d(mix_out), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        ## 特徵蒸餾
        # --- 新增：Bottleneck 層，用於精煉 MixHop 特徵 ---
        (Linear(mix_out, hidden_channels), 'x -> x'),
        # (BatchNorm1d(hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),

        ## aggr lstm, max
        (SAGEConv(hidden_channels, hidden_channels, normalize=True), 'x, edge_index -> x'),  
        (BatchNorm1d(hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (Linear(hidden_channels, out_channels), 'x -> x'),
        (LogSoftmax(dim=1), 'x -> x'),
    ])


# =================================================================================
# GIN 通用框架 vs. GIN專用框架 (因為GIN已經有MLP(batch, ReLU, dropout))
# AI: GIN專用框架能減少梯度消失的風險
# --------- done ---------
# 兩者表現相似，GIN專用框架的版本在某些指標上略微優於通用框架，但差異不大
# nobrd version 的表現在precision下降，說明誤判的情況增加了，不好
# 因此選擇有BRD的版本
# =================================================================================
def MixHopGINModel(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    powers = best_params.get('powers', [0, 1, 2])
    mix_out = hidden_channels * len(powers)

    return Sequential('x, edge_index', [
        
        (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm1d(mix_out), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (GINConv(MLP([mix_out, hidden_channels, hidden_channels], 
                 dropout=dropout, norm="batch_norm")), 'x, edge_index -> x'),
        (BatchNorm1d(hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (Linear(hidden_channels, out_channels), 'x -> x'),
        (LogSoftmax(dim=1), 'x -> x'),
    ])

# def MixHopGINModel_noBRD(in_channels, hidden_channels, out_channels, dropout=0.5, powers=[0, 1, 2]):
#     mix_out = hidden_channels * len(powers)

#     return Sequential('x, edge_index', [
        
#         (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
#         (BatchNorm1d(mix_out), 'x -> x'),
#         (ReLU(inplace=True), 'x -> x'),
#         (Dropout(p=dropout), 'x -> x'),

#         # 注意：GINConv 本身會處理鄰居聚合，內部的 gin_mlp 負責特徵轉換
#         (GINConv(MLP([mix_out, hidden_channels, hidden_channels], 
#                  dropout=dropout)), 'x, edge_index -> x'),
#         # (BatchNorm1d(hidden_channels), 'x -> x'),
#         # (ReLU(inplace=True), 'x -> x'),
#         # (Dropout(p=dropout), 'x -> x'),

#         (Linear(hidden_channels, out_channels), 'x -> x'),
#         (LogSoftmax(dim=1), 'x -> x'),
    
#     ])

