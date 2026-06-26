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
import utils

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Linear_Feature_Extraction(nn.Module):
    def __init__(self, feature, hidden_dim=64):
        super(Linear_Feature_Extraction, self).__init__()
        self.feature = feature
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(self.feature, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
            x = self.linear1(self.act(x))
            x = self.linear2(self.act(x))
            x = self.linear3(self.act(x))
            x = x.squeeze(dim=2)
        return x

class Graph_Feature_Extraction(nn.Module):
    def __init__(self, feature=8, hidden_dim=64, output_dim=1):
        super(Graph_Feature_Extraction, self).__init__()
        self.gnn1 = gnn.GCNConv(feature, hidden_dim)
        self.gnn2 = gnn.GCNConv(hidden_dim, output_dim)
        self.gnn3 = gnn.GATConv(feature, hidden_dim)
        self.gnn4 = gnn.GATConv(hidden_dim, output_dim)

    def forward(self, x):
        v_path = utils.variable_path(x.shape[1], 'all')
        edge_index, edge_weight = utils.tran_adm_to_edge_index(v_path)
        edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
        B, N, T = x.shape  # whole_len, feature, seq_len
        x = x.permute(0, 2, 1)  # → [whole_len, seq_len, feature]
        x = x.reshape(-1, N)  # → [whole_len*seq_len, feature]

        h = self.gnn3(x, edge_index, edge_weight)
        h = self.gnn4(h, edge_index, edge_weight)  # → [whole_len*seq_len, 1]

        # np.save('./result/weight.npy',edge_weight.detach().cpu().numpy())
        # h = self.gnn3(x, self.edge_index)
        # h = self.gnn4(h, self.edge_index)

        h = h.reshape(B, T)
        return h

class AMGNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, path_type=None):
        super(AMGNNLayer, self).__init__()
        self.in_dim = input_dim
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = gnn.SAGEConv(hidden_dim, output_dim)
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = gnn.GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, location):
        if location == 'front':
            out = self.sage1(x, edge_index)
            out = self.sage2(out, edge_index)
            out = self.sage2(out, edge_index)
        if location == 'back':
            out = self.sage2(x, edge_index)
            out = self.sage2(out, edge_index)
            out = self.sage3(out, edge_index)
        return out


class Twodimension_TCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Twodimension_TCNLayer, self).__init__()
        self.drop = nn.Dropout()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        kr, pd = (3, 3), (1, 1)
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn3 = nn.Conv2d(hidden_dim, output_dim, kernel_size=kr, padding=pd)

    def forward(self, x):
        x_0 = x
        h = x_0.unsqueeze(0).unsqueeze(0)

        out = self.cnn1(h)
        out = self.cnn2(self.drop(out))
        out = self.cnn3(self.drop(out))

        out = out.squeeze(0).squeeze(0)
        out = out + x_0
        out = self.linear(out)
        return out


class Double_TCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Double_TCNLayer, self).__init__()
        self.drop = nn.Dropout()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        kr, pd = (3, 3), (1, 1)
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn3 = nn.Conv2d(hidden_dim, output_dim, kernel_size=kr, padding=pd)

    def forward(self, h):
        h_0 = h
        h = h.unsqueeze(0).unsqueeze(0)
        out = self.cnn1(h)

        out_0 = out
        out = self.cnn2(self.drop(out))
        out = self.cnn2(self.drop(out))
        out = out + out_0

        out_1 = out
        out = self.cnn2(self.drop(out))
        out = self.cnn2(self.drop(out))
        out = out + out_1

        out = self.cnn3(self.drop(out))
        h = out.squeeze(0).squeeze(0)

        out = self.linear(h + h_0)
        return out

class TCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, dropout=0.2):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size, padding=kernel_size//2, dilation=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(self.dropout(out))

        out = self.conv2(out)
        out = self.relu(self.dropout(out))

        out = self.conv3(out)
        out = self.dropout(out)

        # 残差和归一化（投影维度匹配）
        if residual.shape != out.shape:
            residual = nn.Conv1d(residual.shape[1], out.shape[1], kernel_size=1)(residual)
        out = out + residual
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.norm(out)
        return out.permute(0, 2, 1)  # [B, C, L]

class  Multipredictor_Aggregator(nn.Module):
    def __init__(self, hidden_dim, pred_dim, type):
        super(Multipredictor_Aggregator, self).__init__()
        self.type = type
        self.num_predictors = 2
        self.pred_dim = pred_dim
        self.weight_layer = nn.Linear(pred_dim * self.num_predictors, self.num_predictors)
        self.fusion = nn.Sequential(
            nn.Linear(2 * pred_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  pred_dim)
        )

    def forward(self, x1, x2, input):
        if self.type == 'min_distance':
            for i in range(self.pred_dim):
                if input.ndim == 3:
                    output = torch.zeros_like(input[:, 0, :])
                    abs1 = torch.abs(input[:, 0, -1] - x1[:, -1])
                    abs2 = torch.abs(input[:, 0, -1] - x2[:, -1])

                    min_abs, min_indices = torch.min(torch.stack([abs1, abs2]), dim=0)
                    # flag = torch.bincount(min_indices.reshape(-1))

                    mask_0 = (min_indices == 0)
                    mask_1 = (min_indices == 1)

                    output[mask_0, -1] = x1[mask_0, -1]
                    output[mask_1, -1] = x2[mask_1, -1]

                if input.ndim == 2:
                    output = torch.zeros_like(input)
                    abs1 = torch.abs(input - x1)
                    abs2 = torch.abs(input - x2)

                    min_abs, min_indices = torch.min(torch.stack([abs1[:, -1], abs2[:, -1]]), dim=0)
                    # flag = torch.bincount(min_indices.reshape(-1))

                    mask_0 = (min_indices == 0)
                    mask_1 = (min_indices == 1)

                    output[mask_0, -1] = x1[mask_0, -1]
                    output[mask_1, -1] = x2[mask_1, -1]

        if self.type == 'softmax':
            preds = [x1, x2]
            concat_preds = torch.cat(preds, dim=-1)  # [B, T * num_predictors]

            weights = self.weight_layer(concat_preds)  # [B, num_predictors]
            weights = F.softmax(weights, dim=-1)  # [B, num_predictors]
            weights = weights.unsqueeze(-1)  # [B, num_predictors, 1]

            # Stack all of the predictors output:[B, num_predictors, T]
            stacked_preds = torch.stack(preds, dim=1)  # [B, num_predictors, T]

            # Weight fusion
            weighted_preds = weights * stacked_preds  # [B, num_predictors, T]
            output = weighted_preds.sum(dim=1)  # [B, T]

        if self.type == 'linear':
            preds = [x1,x2]
            concat = torch.cat(preds, dim=-1)  # [B, T * num_predictors]
            output = self.fusion(concat)  # [B, T]
        return output