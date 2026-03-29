## GNN model ZOO

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE, Sequential
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP, APPNP, ChebConv, GCN2Conv, MixHopConv, BatchNorm

from torch.nn import ReLU, Dropout, BatchNorm1d, Softmax, LogSoftmax, Linear

# =================================================================================
# basic GNNs - done
# =================================================================================
def GCN_Model(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    jk = best_params.get('jk', None)
    return GCN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', jk=jk)

def GAT_Model(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    heads = best_params.get('heads', 8)
    jk = best_params.get('jk', None)
    return GAT(in_channels, hidden_channels, num_layers, out_channels, dropout, heads=heads, norm='batch_norm', jk=jk)

def GraphSAGE_Model(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    jk = best_params.get('jk', None)
    return GraphSAGE(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', jk=jk)

def GIN_Model(best_params=None):
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.5)
    jk = best_params.get('jk', None)
    return GIN(in_channels, hidden_channels, num_layers, out_channels, dropout, norm='batch_norm', jk=jk)

# =================================================================================
# APPNP 
# =================================================================================
def APPNP_Model(best_params=None):
    #### Official PyG APPNP
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    K = best_params.get('K', 10)
    alpha = best_params.get('alpha', 0.1)

    return Sequential('x, edge_index', [
        (Dropout(p=dropout), 'x -> x'),
        (Linear(in_channels, hidden_channels), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),
        (Linear(hidden_channels, out_channels), 'x -> x'),
        (APPNP(K=K, alpha=alpha), 'x, edge_index -> x'),
        # (LogSoftmax(dim=1), 'x -> x'), cancel logsoftmax for focal loss
    ])

# =================================================================================
# ChebNet
# =================================================================================
def ChebNet_Model(best_params=None):
    # Official PyG chebnet
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 16)
    out_channels = best_params.get('out_channels', 2)
    dropout = best_params.get('dropout', 0.5)
    K = best_params.get('K', 3)

    return Sequential('x, edge_index', [
        (ChebConv(in_channels, hidden_channels, K=K), 'x, edge_index -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(dropout), 'x -> x'),
        (ChebConv(hidden_channels, out_channels, K=K), 'x, edge_index -> x'),
        # (LogSoftmax(dim=1), 'x -> x'), cancel logsoftmax for focal loss
    ])

# =================================================================================
# MixHop 
# =================================================================================
def MixHop_Model(best_params=None):
    # Official PyG Mixhop
    in_channels = best_params.get('in_channels', None)
    hidden_channels = best_params.get('hidden_channels', 64)
    out_channels = 2
    dropout = best_params.get('dropout', 0.5)
    powers = best_params.get('powers', "[0, 1, 2]")
    powers = eval(powers)
    current_dim = hidden_channels * len(powers)

    return Sequential('x, edge_index', [
        (Dropout(p=dropout), 'x -> x'),
        
        (MixHopConv(in_channels, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm(current_dim), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),
        
        (MixHopConv(current_dim, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm(current_dim), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (MixHopConv(current_dim, hidden_channels, powers=powers), 'x, edge_index -> x'),
        (BatchNorm(current_dim), 'x -> x'),
        (ReLU(inplace=True), 'x -> x'),
        (Dropout(p=dropout), 'x -> x'),

        (Linear(current_dim, out_channels), 'x -> x'),
    ])