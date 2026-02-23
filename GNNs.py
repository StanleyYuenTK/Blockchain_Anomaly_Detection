import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, MLP, APPNP, ChebConv, GCN2Conv, MixHopConv

from torch.nn import ReLU, Dropout, BatchNorm1d, Softmax, LogSoftmax


def APPNPModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=10, alpha=0.1):
    return Sequential('x, edge_index', [
        ##    input layer, input layer ->   hidden layer
        (MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout), 'x -> x'),
       
        ##    hidden layer,    hidden layer -> output layer
        (MLP([hidden_channels, out_channels], norm=None), 'x -> x'),

        # output layer
        (APPNP(K=K, alpha=alpha, dropout=dropout), 'x, edge_index -> x'),
        (LogSoftmax(dim=1), 'x -> x')

    ])

def ChebNetModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
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

        ## output layer
        (ChebConv(hidden_channels, out_channels, K=K), 'x, edge_index -> x'),
        (LogSoftmax(dim=1), 'x -> x'),
    ])

def MixHopGCNModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
    return Sequential('x, edge_index', [

    ])

def MixHopGATModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
    return Sequential('x, edge_index', [

    ])

def MixHopGraphSAGEModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
    return Sequential('x, edge_index', [

    ])

def MixHopGINModel(in_channels, hidden_channels, out_channels, dropout=0.5, K=3):
    return Sequential('x, edge_index', [

    ])