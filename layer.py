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
    """
    [DEPRECATED for MTTGNet v2 — kept for backward compatibility with old models]
    Aggregates [B, T] scalar predictions from two branches.
    """
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


# ==================== MTTGNet v2 Modules ====================

class VariableInteraction(nn.Module):
    """
    Learnable variable interaction via graph convolution.
    Builds a learned adjacency matrix over F variable nodes,
    then applies GCNConv (with learned edge weights) at each time step
    to fuse variable features.

    Input:  [B, F, T]  — B samples, F variables, T time steps
    Output: [B, T, D]  — variable-fused features, time dim preserved
    """
    def __init__(self, num_vars=8, hidden_dim=64):
        super(VariableInteraction, self).__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        embed_dim = hidden_dim

        # Learnable source and destination embeddings for asymmetric adjacency
        self.src_emb = nn.Parameter(torch.randn(num_vars, embed_dim) * 0.1)
        self.dst_emb = nn.Parameter(torch.randn(num_vars, embed_dim) * 0.1)

        # Initial projection: [1] → [D] per variable value
        self.var_proj = nn.Linear(1, hidden_dim)

        # GCN layers for variable interaction (with learnable edge weights)
        self.gcn1 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, F, T]
        B, F, T = x.shape
        device = x.device

        # Build learnable asymmetric adjacency: [F, F]
        # src_emb @ dst_emb.T models how variable j influences variable i
        adj_raw = torch.matmul(self.src_emb, self.dst_emb.T)  # [F, F]
        adj = torch.softmax(adj_raw, dim=-1)

        # Convert adjacency to edge_index and differentiable edge_weight
        # Build edge_index once from detached adjacency (topology is fixed,
        # but edge weights carry gradients through the softmax)
        with torch.no_grad():
            adj_detached = adj.detach()
            edge_index_np, _ = utils.tran_adm_to_edge_index(
                adj_detached.cpu().numpy())
            edge_index = edge_index_np.to(device)

        # Build differentiable edge weights from adj
        u, v = torch.nonzero(adj_detached > 1e-8, as_tuple=True)
        edge_weight = adj[u, v]  # differentiable! [E]

        # Reshape: each (time_step, variable) is a graph node
        # [B, F, T] → [B, T, F] → [B*T*F, 1]
        x = x.permute(0, 2, 1)  # [B, T, F]
        x = x.reshape(B * T * F, 1)  # [B*T*F, 1]
        x = self.var_proj(x).squeeze(-1)  # [B*T*F, D]

        # Batch the variable graph: one F-node graph per (B*T) time steps
        batched_edge = utils.batch_edge_index(edge_index, B * T, F)
        batched_weight = edge_weight.repeat(B * T)

        # GCNConv on disjoint union of (B*T) variable graphs
        h = self.gcn1(x, batched_edge, batched_weight)  # [B*T*F, D]
        h = self.norm1(h)
        h = self.act(h)
        h = self.gcn2(h, batched_edge, batched_weight)  # [B*T*F, D]
        h = self.norm2(h)

        # Aggregate variable nodes per time step: mean pooling
        h = h.reshape(B * T, F, self.hidden_dim)  # [B*T, F, D]
        h = h.mean(dim=1)  # [B*T, D] — one vector per time step

        # Reshape back: [B*T, D] → [B, T, D]
        h = h.reshape(B, T, self.hidden_dim)
        return h


class TemporalPositionEncoding(nn.Module):
    """
    Dual-scale temporal position encoding:
    - Absolute PE: position within the 60-step lookback window
    - Annual PE: day-of-year encoding for seasonal cycle
    Both are added (not concatenated) to the input features.

    Input:  x [B, T, D], doy_indices [B, T] (optional)
    Output: x + pe [B, T, D]
    """
    def __init__(self, seq_len=60, hidden_dim=64):
        super(TemporalPositionEncoding, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Absolute position encoding (learnable, shared across samples)
        self.abs_pe = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

    def forward(self, x, doy_indices=None):
        # x: [B, T, D]
        B, T, D = x.shape
        # Absolute PE
        out = x + self.abs_pe[:, :T, :]

        # Annual PE (day-of-year encoding)
        if doy_indices is not None:
            annual_pe = utils.day_of_year_encoding(doy_indices, D).to(x.device)
            out = out + annual_pe * 0.1  # scale down to avoid overwhelming features

        return out


class ResidualConv1D(nn.Module):
    """
    Simple 2-layer 1D convolution with residual connection.
    Replaces the buggy Twodimension_TCNLayer (which used 2D conv on batch dim).

    Input:  [B, T, D]
    Output: [B, T, D] (same shape)
    """
    def __init__(self, channels=64, kernel_size=3, dropout=0.2):
        super(ResidualConv1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2, padding_mode='replicate')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2, padding_mode='replicate')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D] → [B, D, T]
        residual = x
        x_t = x.permute(0, 2, 1)  # [B, D, T]
        out = self.conv1(x_t)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out + x_t  # residual
        return out.permute(0, 2, 1)  # [B, T, D]


class MultiStepPredictionHead(nn.Module):
    """
    MLP prediction head: takes the last time step's representation
    and projects to multiple future steps.

    Input:  h [B, D] — representation at the last time step
    Output: [B, pred_len]
    """
    def __init__(self, input_dim=64, hidden_dim=128, pred_len=4, dropout=0.1):
        super(MultiStepPredictionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, h):
        # h: [B, D]
        return self.mlp(h)  # [B, pred_len]


class BranchAggregator(nn.Module):
    """
    Aggregates two branch outputs (each [B, D]) into a single representation.
    Supports three fusion modes: softmax, linear, min_distance.

    Input:  h1 [B, D], h2 [B, D]
    Output: [B, D]
    """
    def __init__(self, hidden_dim=64, agg_type='softmax'):
        super(BranchAggregator, self).__init__()
        self.agg_type = agg_type
        self.hidden_dim = hidden_dim
        self.num_branches = 2

        # Softmax: learnable weight per branch
        self.weight_layer = nn.Linear(hidden_dim * self.num_branches, self.num_branches)

        # Linear: project concatenated features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_branches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h1, h2):
        # h1, h2: [B, D]
        if self.agg_type == 'softmax':
            concat = torch.cat([h1, h2], dim=-1)  # [B, 2D]
            weights = self.weight_layer(concat)  # [B, 2]
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # [B, 2, 1]
            stacked = torch.stack([h1, h2], dim=1)  # [B, 2, D]
            output = (weights * stacked).sum(dim=1)  # [B, D]

        elif self.agg_type == 'linear':
            concat = torch.cat([h1, h2], dim=-1)  # [B, 2D]
            output = self.fusion(concat)  # [B, D]

        elif self.agg_type == 'mean':
            output = (h1 + h2) / 2.0

        else:
            raise ValueError(f"Unknown agg_type: {self.agg_type}. "
                             f"Choose from ['softmax', 'linear', 'mean'].")

        return output