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
import utils

class dayGNNnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_weight, feature):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_weight = nn.Parameter(edge_weight)
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = gnn.SAGEConv(hidden_dim, output_dim)
        kr, pd = (3, 3), (1, 1)
        self.cnn1 = nn.Conv2d(1, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn3 = nn.Conv2d(hidden_dim, 1, kernel_size=kr, padding=pd)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.cnn = nn.Conv2d(1,1,kernel_size=(2,2),padding=(0,1))
        self.linear1 = nn.Linear(feature, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = x.squeeze(dim=2)
        h = self.sage1(x, edge_index)
        h = self.sage2(h, edge_index)
        h = self.sage2(h, edge_index)

        out_1 = h
        h = h.unsqueeze(0).unsqueeze(0)
        out = self.cnn1(h)
        out = self.cnn2(self.drop(out))
        out = self.cnn3(self.drop(out))
        out = out.squeeze(0).squeeze(0)
        out = out + out_1

        out = self.linear2(out)
        h = self.sage2(out, edge_index)
        h = self.sage2(h, edge_index)
        h = self.sage3(h, edge_index)
        return h

class yearGNNnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_weight, feature):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_weight = nn.Parameter(edge_weight)
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = gnn.SAGEConv(hidden_dim, output_dim)
        kr, pd = (3, 3), (1, 1)
        self.cnn1 = nn.Conv2d(1, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kr, padding=pd)
        self.cnn3 = nn.Conv2d(hidden_dim, 1, kernel_size=kr, padding=pd)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.linear1 = nn.Linear(feature, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = x.squeeze(dim=2)
            # np.save('year_embedding.npy', x.detach().cpu().numpy())
        h = self.sage1(x, edge_index)
        h = self.sage2(h, edge_index)
        h = self.sage2(h, edge_index)

        out_1 = h
        h = h.unsqueeze(0).unsqueeze(0)
        out = self.cnn1(h)
        out = self.cnn2(self.drop(out))
        out = self.cnn3(self.drop(out))
        out = out.squeeze(0).squeeze(0)
        out = out + out_1

        out = self.linear2(out)
        h = self.sage2(out, edge_index)
        h = self.sage2(h, edge_index)
        h = self.sage3(h, edge_index)
        return h

class MTTGnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, w1, w2, feature):
        super(MTTGnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.m1 = dayGNNnet(self.input_dim, self.hidden_dim, self.output_dim, self.w1, feature)
        self.m2 = yearGNNnet(self.input_dim, self.hidden_dim, self.output_dim, self.w2, feature)

    def forward(self, input, idx1, idx3):
        x1 = self.m1(input, idx1)
        x2 = self.m2(input, idx3)

        # Multipredictor Aggregator
        for i in range(self.output_dim):
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
        return output

class DayGNNnet(nn.Module):
    def __init__(self, w_daily, feature=8, input_dim=60, hidden_dim=64, output_dim=1):
        super(DayGNNnet, self).__init__()
        self.w_daily = nn.Parameter(w_daily)

        self.linear_feature_extractor = layer.Linear_Feature_Extraction(feature=feature, hidden_dim=hidden_dim)
        self.graph_feature_extractor = layer.Graph_Feature_Extraction(feature=feature, hidden_dim=hidden_dim, output_dim=1)

        self.amgnn = layer.AMGNNLayer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        # self.tcn1 = layer.TCNLayer(input_dim=1, hidden_dim=64, output_dim=1)
        self.tcn2 = layer.Twodimension_TCNLayer(input_dim=1, hidden_dim=64, output_dim=1)
        # self.tcn3 = layer.Double_TCNLayer(input_dim=1, hidden_dim=64, output_dim=1)

    def forward(self, x, edge_index=None):
        # x_embed = self.linear_feature_extractor(x)
        x_embed = self.graph_feature_extractor(x)
        x_amgnn1 = self.amgnn(x_embed, edge_index, 'front')
        x_tcn = self.tcn2(x_amgnn1)
        x_amgnn2 = self.amgnn(x_tcn, edge_index, 'back')
        return x_amgnn2


class YearGNNnet(nn.Module):
    def __init__(self, w_yearly, feature=8, input_dim=60, hidden_dim=64, output_dim=1):
        super(YearGNNnet, self).__init__()
        self.w_yearly = nn.Parameter(w_yearly)

        self.linear_feature_extractor = layer.Linear_Feature_Extraction(feature=feature, hidden_dim=hidden_dim)
        self.graph_feature_extractor = layer.Graph_Feature_Extraction(feature=feature, hidden_dim=hidden_dim, output_dim=1)

        self.amgnn = layer.AMGNNLayer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        # self.tcn1 = layer.TCNLayer(input_dim=1, hidden_dim=64, output_dim=1)
        self.tcn2 = layer.Twodimension_TCNLayer(input_dim=1, hidden_dim=64, output_dim=1)
        # self.tcn3 = layer.Double_TCNLayer(input_dim=1, hidden_dim=64, output_dim=1)

    def forward(self, x, edge_index=None):
        # x_embed = self.linear_feature_extractor(x)
        x_embed = self.graph_feature_extractor(x)
        x_amgnn1 = self.amgnn(x_embed, edge_index, 'front')
        x_tcn = self.tcn2(x_amgnn1)
        x_amgnn2 = self.amgnn(x_tcn, edge_index, 'back')
        return x_amgnn2

class MTTGNet(nn.Module):
    """
    [DEPRECATED — use MTTGNetv2 instead]
    Kept for backward compatibility with old checkpoints.
    """
    def __init__(self, w_daily, w_yearly, feature, input_dim=60, hidden_dim=64, output_dim=1, aggregator_type='softmax'):
        super(MTTGNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature = feature

        self.w_daily = nn.Parameter(w_daily)
        self.w_yearly = nn.Parameter(w_yearly)

        self.daygnn = DayGNNnet(self.w_daily, self.feature, self.input_dim, self.hidden_dim, self.output_dim)
        self.yeargnn = YearGNNnet(self.w_yearly, self.feature, self.input_dim, self.hidden_dim, self.output_dim)

        self.aggregator = layer.Multipredictor_Aggregator(hidden_dim=hidden_dim, pred_dim=output_dim, type=aggregator_type)

    def forward(self, x, index_daily, index_yearly):
        x_daily = self.daygnn(x, index_daily)
        x_yearly = self.yeargnn(x, index_yearly)
        output = self.aggregator(x1=x_daily, x2=x_yearly, input=x)
        return output


# ==================== MTTGNet v2 Architecture ====================

class DayBranch(nn.Module):
    """
    Day-scale branch: models short-range weather dynamics (hours to days).
    - Variable interaction: learnable adjacency over 8 atmospheric variables
    - Temporal graph: edges i↔i+1 (adjacent 6h), i↔i+4 (same time next day)
    - Convolution: ResidualConv1D for local pattern extraction
    """
    def __init__(self, num_vars=8, hidden_dim=64, seq_len=60):
        super(DayBranch, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.var_interact = layer.VariableInteraction(num_vars, hidden_dim)
        self.pos_enc = layer.TemporalPositionEncoding(seq_len, hidden_dim)
        self.amgnn_front = layer.AMGNNLayer(hidden_dim, hidden_dim, hidden_dim)
        self.conv = layer.ResidualConv1D(hidden_dim)
        self.amgnn_back = layer.AMGNNLayer(hidden_dim, hidden_dim, hidden_dim)

        # Day temporal graph: period=4 (24-hour cycle at 6-hourly sampling)
        self.register_buffer(
            'temporal_graph',
            utils.build_temporal_graph(seq_len, period_steps=4, self_loop=True)
        )

    def forward(self, x, doy_indices=None):
        # x: [B, F, T]
        B, _, T = x.shape

        # 1. Variable interaction
        h = self.var_interact(x)  # [B, T, D]

        # 2. Position encoding
        h = self.pos_enc(h, doy_indices)  # [B, T, D]

        # 3. AMGNN (front) — batch the temporal graph
        batched_edges = utils.batch_edge_index(self.temporal_graph, B, T)
        h = self.amgnn_front(h.reshape(B * T, self.hidden_dim), batched_edges, 'front')
        h = h.reshape(B, T, self.hidden_dim)

        # 4. Temporal convolution
        h = self.conv(h)  # [B, T, D]

        # 5. AMGNN (back)
        batched_edges = utils.batch_edge_index(self.temporal_graph, B, T)
        h = self.amgnn_back(h.reshape(B * T, self.hidden_dim), batched_edges, 'back')
        h = h.reshape(B, T, self.hidden_dim)

        return h  # [B, T, D]


class YearBranch(nn.Module):
    """
    Year-scale branch: models long-range climate dynamics (weeks to seasons).
    - Variable interaction: separate learnable adjacency (different from DayBranch)
    - Temporal graph: edges i↔i+1, i↔i+28 (~weekly context)
    - Convolution: ResidualConv1D (same structure as DayBranch; differentiation
      comes from the wider-period temporal graph and independent variable adjacency)
    """
    def __init__(self, num_vars=8, hidden_dim=64, seq_len=60):
        super(YearBranch, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.var_interact = layer.VariableInteraction(num_vars, hidden_dim)
        self.pos_enc = layer.TemporalPositionEncoding(seq_len, hidden_dim)
        self.amgnn_front = layer.AMGNNLayer(hidden_dim, hidden_dim, hidden_dim)
        self.conv = layer.ResidualConv1D(hidden_dim)
        self.amgnn_back = layer.AMGNNLayer(hidden_dim, hidden_dim, hidden_dim)

        # Year temporal graph: period=28 (~weekly context at 6-hourly sampling)
        self.register_buffer(
            'temporal_graph',
            utils.build_temporal_graph(seq_len, period_steps=28, self_loop=True)
        )

    def forward(self, x, doy_indices=None):
        # x: [B, F, T]
        B, _, T = x.shape

        # 1. Variable interaction
        h = self.var_interact(x)  # [B, T, D]

        # 2. Position encoding
        h = self.pos_enc(h, doy_indices)  # [B, T, D]

        # 3. AMGNN (front)
        batched_edges = utils.batch_edge_index(self.temporal_graph, B, T)
        h = self.amgnn_front(h.reshape(B * T, self.hidden_dim), batched_edges, 'front')
        h = h.reshape(B, T, self.hidden_dim)

        # 4. Temporal convolution
        h = self.conv(h)  # [B, T, D]

        # 5. AMGNN (back)
        batched_edges = utils.batch_edge_index(self.temporal_graph, B, T)
        h = self.amgnn_back(h.reshape(B * T, self.hidden_dim), batched_edges, 'back')
        h = h.reshape(B, T, self.hidden_dim)

        return h  # [B, T, D]


class MTTGNetv2(nn.Module):
    """
    Multivariate Multi-period Temperature Time Series Graph Neural Network v2.

    Architecture:
    1. DayBranch: short-range weather dynamics (daily cycle)
    2. YearBranch: long-range climate dynamics (seasonal context)
    3. BranchAggregator: fuse the two branch representations
    4. MultiStepPredictionHead: project to multiple future steps

    Args:
        num_vars: number of input variables (default 8)
        hidden_dim: hidden dimension (default 64)
        seq_len: lookback sequence length (default 60)
        pred_len: prediction horizon steps (default 4 = 24 hours)
        aggregator_type: fusion mode ['softmax', 'linear', 'mean']
    """
    def __init__(self, num_vars=8, hidden_dim=64, seq_len=60, pred_len=4,
                 aggregator_type='softmax'):
        super(MTTGNetv2, self).__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.day_branch = DayBranch(num_vars, hidden_dim, seq_len)
        self.year_branch = YearBranch(num_vars, hidden_dim, seq_len)
        self.aggregator = layer.BranchAggregator(hidden_dim, aggregator_type)
        self.pred_head = layer.MultiStepPredictionHead(hidden_dim, 128, pred_len)

    def forward(self, x, doy_indices=None):
        """
        Args:
            x: [B, F, T] — input time series batch
            doy_indices: [B, T] — day-of-year indices (optional, for seasonal PE)

        Returns:
            output: [B, pred_len] — multi-step predictions
        """
        # 1. Process through day and year branches
        h_day = self.day_branch(x, doy_indices)    # [B, T, D]
        h_year = self.year_branch(x, doy_indices)  # [B, T, D]

        # 2. Take representation at the last time step
        h_day_last = h_day[:, -1, :]    # [B, D]
        h_year_last = h_year[:, -1, :]  # [B, D]

        # 3. Aggregate the two branch representations
        h_fused = self.aggregator(h_day_last, h_year_last)  # [B, D]

        # 4. Multi-step prediction
        output = self.pred_head(h_fused)  # [B, pred_len]

        return output
