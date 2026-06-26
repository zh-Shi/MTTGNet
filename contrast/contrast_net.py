import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as gnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random
import os
import math
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
import layer

class Contrast_GNNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_weight, gnn_style):
        super(Contrast_GNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_weight = nn.Parameter(edge_weight)
        self.gnn_style = gnn_style
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, output_dim)
        self.che1 = gnn.ChebConv(input_dim, hidden_dim, K=3)
        self.che2 = gnn.ChebConv(hidden_dim, output_dim, K=3)
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, output_dim)
        self.gin1 = gnn.GraphConv(input_dim, hidden_dim)
        self.gin2 = gnn.GraphConv(hidden_dim, output_dim)
        self.tran1 = gnn.TransformerConv(input_dim, hidden_dim)
        self.tran2 = gnn.TransformerConv(hidden_dim, output_dim)
        self.tag1 = gnn.TAGConv(input_dim, hidden_dim)
        self.tag2 = gnn.TAGConv(hidden_dim, output_dim)
        self.gat1 = gnn.GATConv(input_dim, hidden_dim)
        self.gat2 = gnn.GATConv(hidden_dim, output_dim)
        self.linear_feature_extractor = layer.Linear_Feature_Extraction(feature=input_dim, hidden_dim=hidden_dim)
        self.tcn = layer.Double_TCNLayer(input_dim=60, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x, edge_index):
        if self.gnn_style == "GCN":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.gcn1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.gcn2(h, edge_index)
        elif self.gnn_style == "Cheb":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.che1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.che2(h, edge_index)
        elif self.gnn_style == "ResGraphnet":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.sage1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.sage2(h, edge_index)
        elif self.gnn_style == "GIN":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.gin1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.gin2(h, edge_index)
        elif self.gnn_style == "UniMP":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.tran1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.tran2(h, edge_index)
        elif self.gnn_style == "TAGCN":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.tag1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.tag2(h, edge_index)
        elif self.gnn_style == "GAT":
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = x.squeeze(dim=2)
            h = self.gat1(x, edge_index)
            h_0 = h
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.linear2(h + h_0)
            h = self.gat2(h, edge_index)
        return h

class Contrast_RNNmodel(nn.Module):
    def __init__(self, rnn_style, in_dim, hid_dim, out_dim, l_x, num_layers):
        super(Contrast_RNNmodel, self).__init__()
        self.rnn_style = rnn_style
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.linear_pre = nn.Linear(in_dim, hid_dim)
        self.lstm1 = nn.LSTM(hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.gru1 = nn.GRU(hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.last1 = nn.Linear(hid_dim, 1)
        self.last2 = nn.Linear(l_x, out_dim)

    def forward(self, x):
        h = x.unsqueeze(2)
        h = self.linear_pre(h)
        if self.rnn_style == "LSTM":
            h, (_, _) = self.lstm1(self.pre(h))
        elif self.rnn_style == "GRU":
            h, (_) = self.gru1(self.pre(h))
        else:
            raise TypeError("Unknown Type of rnn_style!")
        h = self.last1(h).squeeze(2)
        h = self.last2(h)
        return h