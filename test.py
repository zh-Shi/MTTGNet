import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random
import os
import utils
import net

x = np.load('./data/ERA5_part/ERA5_temp/era5_t2m_daily_1956_2023_mean_anomaly.npy')
num = x.shape[0]

# 导入其他数据
data = np.load('./data/ERA5_part/nontemp_data_process/data_mixed.npy')
x = np.vstack((x, data))
np.save("data_all.npy", x.T)