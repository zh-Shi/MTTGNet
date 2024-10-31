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
from sklearn.metrics import r2_score
from torch.utils.data import Dataset


#输出数据
def plot_data(data):
    l = len(data)
    plt.figure(figsize=(120, 8))
    plt.plot(data,c = "orangered",alpha = 0.6)
    plt.yticks(np.arange(int(min(data)), int(max(data)+1), 0.5), fontsize = 20)
    date = ['1956_01_01', '1969_08_10', '1983_03_19', '1996_10_28', '2010_06_07', '2023_12_31']
    plt.xticks([0, int(0.2*l), int(0.4*l), int(0.6*l), int(0.8*l), l-1], date, fontsize = 20)
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("t2m(°C)",fontsize = 20)
    plt.show()

#给出一定size下的布尔变量，某个index（索引）为True，其余为False，例:index_to_mask(3,5)为[False, False, False,  True, False]
def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

#创建路径图的邻接矩阵（有向图）
def path_graph(m,time_style):
    if time_style == 'day':
        adm = np.zeros(shape=(m, m))
        for i in range(m - 1):
            adm[i, i + 1] = 1
        return adm
    if time_style == 'year':
        adm = np.zeros(shape=(m, m))
        for i in range(m - 366):
            adm[i, i + 1] = 1
            adm[i, i + 365] = 1
        return adm
    if time_style == 'month_year':
        adm = np.zeros(shape=(m, m))
        for i in range(m - 31):
            adm[i, i + 1] = 1
            adm[i, i + 12] = 1
        return adm

#将邻接矩阵转换为边的索引和权重，adm即为邻接矩阵的缩写
def tran_adm_to_edge_index(adm):
    if isinstance(adm, np.ndarray):#判断输入是否为NumPy数组
        u, v = np.nonzero(adm)#返回adm中非0元素的索引，表示存在边的位置a
        num_edges = u.shape[0]#边的数量
        edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])#将u和v合并成一个边的索引
        edge_weight = np.zeros(shape=u.shape)#初始化边的权重数组
        for i in range(num_edges):
            edge_weight_one = adm[u[i], v[i]]#获取对应位置的权重
            edge_weight[i] = edge_weight_one
        edge_index = torch.from_numpy(edge_index.astype(np.int64))#转换为Tensor张量
        edge_weight = torch.from_numpy(edge_weight.astype(np.float32))
        return edge_index, edge_weight
    elif torch.is_tensor(adm):#判断输入是否为Tensor张量，其它操作同上但使用的代码不同
        u, v = torch.nonzero(adm, as_tuple=True)
        num_edges = u.shape[0]
        edge_index = torch.cat((u.view(1, -1), v.view(1, -1)))
        edge_weight = torch.zeros(size=u.shape)
        for i in range(num_edges):
            edge_weight[i] = adm[u[i], v[i]]
        return edge_index, edge_weight

#求r2_score（且不区分类型和维数）
def get_r2_score(y1, y2, axis):
    if (type(y1) is np.ndarray) & (type(y2) is np.ndarray):#numpy数组类型
        pass
    elif (torch.is_tensor(y1)) & (torch.is_tensor(y2)):#pytorch张量类型
        y1 = y1.detach().cpu().numpy()#去除张量特性将其转换成numpy数组
        y2 = y2.detach().cpu().numpy()
    if y1.ndim == 1:#如果y1是1维数组，则添加一个维度
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
    elif y1.ndim == 2:
        pass
    if axis == 0:#如果是第一维
        num_col = y1.shape[0]
    elif axis == 1:
        num_col = y1.shape[1]
    r2_all = 0
    for i in range(num_col):
        if axis == 0:
            y1_one = y1[i, :]
            y2_one = y2[i, :]
        elif axis == 1:
            y1_one = y1[:, i]
            y2_one = y2[:, i]
        r2_one = r2_score(y2_one, y1_one)#计算单列/行的R2分数
        r2_all = r2_all + r2_one
    r2 = r2_all / num_col
    return r2

def get_rmse(a1, a2):
    a1, a2 = a1.reshape(-1), a2.reshape(-1)
    m_1, m_2 = a1.shape[0], a2.shape[0]
    if m_1 < m_2:
        m = m_1
    else:
        m = m_2
    a1_m, a2_m = a1[:m], a2[:m]
    result = np.sqrt(np.sum(np.square(a1_m - a2_m)) / m)
    return result

#创建输入和输出序列（此处的x_length设置为60，即图信号为60通道）
def create_inout_sequences(input_data, x_length=60, y_length=1, ml_dim=0, ld1=True):#此处设置了若干默认数值
    seq_list, seq_arr, label_arr = [], None, None
    data_length = input_data.shape[input_data.ndim-1]
    x_y_length = x_length + y_length
    seq_arr = np.zeros(data_length - x_y_length, x_length)
    label_arr = np.zeros()
    for i in range(data_length - x_y_length + 1):
        if input_data.ndim == 2:
            seq = input_data[:,i: (i + x_length)]
            label = input_data[ml_dim, (i + x_length): (i + x_length + y_length)].reshape(1, -1)
            seq = np.expand_dims(seq, 0)
        elif input_data.ndim == 1:
            seq = input_data[i: (i + x_length)]
            label = input_data[(i + x_length): (i + x_length + y_length)]
            seq, label = seq.reshape(1, -1), label.reshape(1, -1)
        if (seq_arr is None) & (label_arr is None):
            seq_arr, label_arr = seq, label
        else:
            seq_arr, label_arr = np.concatenate([seq_arr, seq], axis=0), np.concatenate([label_arr, label], axis=0)
    return seq_arr, label_arr

