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

def normalized(data):
    mx1 = np.max(data)
    mn1 = np.min(data)
    mm1 = mx1 - mn1
    mn1 = mn1 * np.ones_like(data)
    data_normalized = ((data - mn1) / mm1)*2-1
    return data_normalized

#输出数据
def plot_data(data):
    l = len(data)
    plt.figure(figsize=(120, 8))
    plt.plot(data, c="red", alpha=0.6)
    # plt.yticks(np.arange(int(min(data)), int(max(data)+1)), fontsize=20)
    date = ['1956_01_01', '1969_08_10', '1983_03_19', '1996_10_28', '2010_06_07', '2023_12_31']
    plt.xticks([0, int(0.2*l), int(0.4*l), int(0.6*l), int(0.8*l), l-1], date, fontsize=20)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("t2m(°C)", fontsize=20)
    plt.show()

#给出一定size下的布尔变量，某个index（索引）为True，其余为False，例:index_to_mask(3,5)为[False, False, False,  True, False]
def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

#创建路径图的邻接矩阵（有向图）
def path_graph(m,time_style):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i, i + 1] = 1
    if time_style == 'none':
        return adm
    if time_style == 'day':
        for i in range(m - 4 - 1):
            adm[i, i + 4] = 1
    if time_style == 'year':
        for i in range(m - 365*4 - 1):
            adm[i, i + 365*4] = 1
    if time_style == 'mix':
        for i in range(m - 365*4 - 1):
            adm[i, i + 4] = 1
            adm[i, i + 365*4] = 1
    return adm

def variable_path(m,type):
    adm = np.zeros(shape=(m, m))
    if type == 'single':
        adm = np.eye(m)
        for i in range(m):
            adm[0, i] = 1
    if type == 'all':
        adm = np.ones(shape=(m, m))
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
    x_y_length = x_length + y_length - 1
    if input_data.ndim == 2:
        seq_arr = np.zeros((data_length - x_y_length, input_data.shape[0], x_length))
        label_arr = np.zeros((data_length - x_y_length, y_length))
        for i in range(data_length - x_y_length):
            seq = input_data[:, i: (i + x_length)]
            label = input_data[ml_dim, (i + x_length): (i + x_length + y_length)].reshape(1, -1)
            seq_arr[i, :, :] = seq
            label_arr[i, :] = label
    elif input_data.ndim == 1:
        seq_arr = np.zeros((data_length - x_y_length, x_length))
        label_arr = np.zeros((data_length - x_y_length, y_length))
        for i in range(data_length - x_y_length):
            seq = input_data[i: (i + x_length)]
            label = input_data[(i + x_length): (i + x_length + y_length)]
            seq, label = seq.reshape(1, -1), label.reshape(1, -1)
            seq_arr[i, :] = seq
            label_arr[i, :] = label
            # seq_arr, label_arr = np.concatenate([seq_arr, seq], axis=0), np.concatenate([label_arr, label], axis=0)
    return seq_arr, label_arr

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def NRMSE(pred, true):
    return np.sqrt(np.mean(np.power((pred - true), 2))) / (np.mean(np.abs(true)))

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def evaluation_all(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    nd = ND(pred,true)
    nrmse = NRMSE(pred,true)

    return mae, mse, rmse, mape, mspe, rse , corr, nd, nrmse

from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
def plot_base2(lats, lons, x1, x2, t1, t2, t, fig_si, fo_si, fo_ti_si, cb_t="", bins=7):
    fig, axes = plt.subplots(1, 2, figsize=fig_si, sharey=True)

    if t is not None:
        fig.suptitle(t, fontsize=fo_si, fontweight='bold')

    x1 = np.where(x1 == 0, np.nan, x1)
    x2 = np.where(x2 == 0, np.nan, x2)
    lat_bounds = [np.min(lats), np.max(lats)]
    lon_bounds = [np.min(lons), np.max(lons)]
    parallels = np.linspace(lat_bounds[0], lat_bounds[1], bins)
    meridians = np.linspace(lon_bounds[0], lon_bounds[1], bins)

    data_list = [(x1, t1, axes[0]), (x2, t2, axes[1])]

    for data, title, ax in data_list:
        ax.set_title(title, fontsize=fo_si)
        m = Basemap(projection='cyl', ax=ax,
                    llcrnrlon=lon_bounds[0], llcrnrlat=lat_bounds[0],
                    urcrnrlon=lon_bounds[1], urcrnrlat=lat_bounds[1])
        # m.drawcoastlines()
        m.drawmapboundary()
        m.drawparallels(np.around(parallels, 0), labels=[1, 0, 0, 0], fontsize=fo_ti_si)
        m.drawmeridians(np.around(meridians, 0), labels=[0, 0, 0, 1], fontsize=fo_ti_si)

        x, y = np.meshgrid(lons, lats)
        data = ma.masked_where(data == 0, data)
        cs = m.contourf(x, y, data, 11, cmap=plt.cm.Spectral_r)


    # 添加一个 colorbar 放在右侧
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # (x起点, y起点, 宽度, 高度)
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_label(cb_t, fontsize=fo_si, labelpad=0)
    cbar.ax.tick_params(labelsize=fo_ti_si)
    cbar.set_label("Temperature (K)", fontsize=fo_si, labelpad=5)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # 适配布局，避免标题遮挡
    plt.show()
    return fig


def standardize(data_series: np.ndarray, mean: float = None, std: float = None) -> tuple[
    np.ndarray, float, float]:
    if not isinstance(data_series, np.ndarray) or data_series.ndim != 1:
        raise ValueError("输入 data_series 必须是一个一维 NumPy 数组。")

    # 如果没有提供均值和标准差，则从当前数据计算
    if mean is None:
        calculated_mean = np.mean(data_series)
    else:
        calculated_mean = mean

    if std is None:
        calculated_std = np.std(data_series)
    else:
        calculated_std = std

    # 避免除以零的情况，如果标准差为零，则所有值都将变为零
    if calculated_std == 0:
        standardized_series = np.zeros_like(data_series, dtype=float)
    else:
        standardized_series = (data_series - calculated_mean) / calculated_std

    return standardized_series, calculated_mean, calculated_std


def inverse_standardize(standardized_series: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not isinstance(standardized_series, np.ndarray) or standardized_series.ndim != 1:
        raise ValueError("输入 standardized_series 必须是一个一维 NumPy 数组。")

    if std == 0:  # 如果原始标准差为0，说明原始数据都是同一个值，逆转换后也应该是这个值
        original_series = np.full_like(standardized_series, mean, dtype=float)
    else:
        original_series = (standardized_series * std) + mean
    return original_series
