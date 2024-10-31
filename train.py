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
from sklearn.preprocessing import MinMaxScaler
import pymannkendall as mk

#固定随机数
def seed_everything(seed):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(233)


#设置超参数
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
l_x = 120
l_y = 12
hidden_dim = 64
lr = 0.001
weight_decay = 0.001
epochs = 3000
ratio_train = 0.9
cut_co2 = False

x = np.load('./data/ERA5_part/ERA5_temp/era5_t2m_daily_1956_2023_mean_anomaly.npy')
num = x.shape[0]

# 导入其他数据
data = np.load('./data/ERA5_part/nontemp_data_process/data_mixed.npy')
x = np.vstack((x, data))

num_train = int(ratio_train * num)
if x.ndim == 2:
    data_train, data_test = x[:, :num_train], x[:, num_train:num]
if x.ndim == 1:
    data_train, data_test = x[:num_train], x[num_train:num]

# 削减co2浓度
if cut_co2:
    x[7, :] = np.concatenate((x[7, :num_train-1000], 2*x[7, num_train-1000]-x[7, num_train-1000:]))

start_time = time.time()
x_train, y_train = utils.create_inout_sequences(data_train, l_x, l_y)
x_test, y_test = utils.create_inout_sequences(data_test, l_x, l_y)
print(time.time()-start_time)

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

num_nodes = x_train.shape[0] + x_test.shape[0]
num_train = x_train.shape[0]
num_test = x_test.shape[0]

x = torch.cat((x_train, x_test), dim=0)
y = torch.cat((y_train, y_test), dim=0)

# adm_day = utils.path_graph(num_nodes, 'day')
# adm_year = utils.path_graph(num_nodes, 'year')
# edge_index_day, edge_weight_day = utils.tran_adm_to_edge_index(adm_day)
# edge_index_year, edge_weight_year = utils.tran_adm_to_edge_index(adm_year)

adm_day = utils.path_graph(num_train, 'day')
adm_year = utils.path_graph(num_train, 'year')
edge_index_day, edge_weight_day = utils.tran_adm_to_edge_index(adm_day)
edge_index_year, edge_weight_year = utils.tran_adm_to_edge_index(adm_year)

adm_day = utils.path_graph(num_test, 'day')
adm_year = utils.path_graph(num_test, 'year')
edge_index_day0, edge_weight_day0 = utils.tran_adm_to_edge_index(adm_day)
edge_index_year0, edge_weight_year0 = utils.tran_adm_to_edge_index(adm_year)

# node1 = np.linspace(0,num_nodes-2,num_nodes-1)
# node2 = np.linspace(1,num_nodes-1,num_nodes-1)
# w = np.ones(num_nodes-1)
#
# edge_index = torch.from_numpy(np.vstack([node1,node2]).astype(np.int64))
# edge_weight = torch.from_numpy(w.astype(np.float32))

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = utils.index_to_mask(train_index, num_nodes).to(device)
test_mask = utils.index_to_mask(test_index, num_nodes).to(device)


model = net.MTTGnet(l_x, hidden_dim, l_y, edge_weight_day, edge_weight_year).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
edge_index_day, edge_index_year = edge_index_day.to(device), edge_index_year.to(device)
edge_index_day0, edge_index_year0 = edge_index_day0.to(device), edge_index_year0.to(device)
#损失记录
para_trainloss = np.zeros(epochs)
para_testloss = np.zeros(epochs)
para_r2_train = np.zeros(epochs)
para_r2_test = np.zeros(epochs)
para_rmse = np.zeros(epochs)
min_rmse = 100

start_time = time.time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output, flag = model(x_train, edge_index_day, edge_index_year)
    flag = flag.detach().cpu().numpy()
    # output = model(x, edge_index_day)
    # output = (output + 1) / 2 * mm1 + mn1
    # output_train, y_train = output[train_mask], y[train_mask]
    output_train = output
    y_train = y[train_mask]
    train_loss = criterion(output_train[:, -1], y_train[:, -1])
    para_trainloss[epoch] = train_loss
    train_loss.backward()
    optimizer.step()

    model.eval()
    # y_test_1 = y[test_mask][:-len_interp-l_y, :]
    output, flag = model(x_test, edge_index_day0, edge_index_year0)
    flag = flag.detach().cpu().numpy()
    y_test_1 = y[test_mask][:- l_y, :]
    y_test_2 = y[test_mask][-l_y:, :]
    y_test = torch.cat((y_test_1, y_test_2), dim=0)
    # output_test = output[test_mask][:-len_interp, :]
    # output_test = output[test_mask][:, :]
    output_test = output
    test_loss = criterion(output_test[:, -1], y_test[:, -1])
    para_testloss[epoch] = test_loss

    train_true = y_train.detach().cpu().numpy()[:, -1]
    train_predict = output_train.detach().cpu().numpy()[:, -1]
    test_true = y_test.detach().cpu().numpy()[:, -1]
    test_predict = output_test.detach().cpu().numpy()[:, -1]
    r2_train = utils.get_r2_score(train_predict, train_true, axis=1)
    r2_test = utils.get_r2_score(test_predict, test_true, axis=1)
    para_r2_train[epoch] = r2_train
    para_r2_test[epoch] = r2_test
    rmse = utils.get_rmse(train_predict, train_true)
    para_rmse[epoch] = rmse
    min_rmse = min(rmse, min_rmse)
    if min_rmse == rmse:
        torch.save(model.state_dict(), 'model.pth')
        ep = epoch
        best_rmse_train = utils.get_rmse(train_predict, train_true)
        best_rmse_test = utils.get_rmse(test_predict, test_true)
        best_r2_train = utils.get_r2_score(train_predict, train_true, axis=1)
        best_r2_test = utils.get_r2_score(test_predict, test_true, axis=1)
        best_flag_day, best_flag_year = flag[0], flag[1]
        if cut_co2:
            np.save('co2_test_predict.npy', test_predict)
        if cut_co2 == False:
            np.save('test_predict.npy', test_predict)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))
        mse = np.mean(np.square(train_predict - train_true))
        print("mse: {:.8f}".format(mse))
        print("dayNet:{}  yearNet:{}".format(flag[0], flag[1]))



end_time = time.time()
run_time = end_time - start_time
print("Run_time:{}".format(run_time))
print('best model epoch:{}'.format(ep))

print("Best:RMSE_Train={:.5f}  RMSE_Test={:.5f}  R2_Train={:.7f}  R2_Test={:.7f}".
      format(best_rmse_train, best_rmse_test, best_r2_train, best_r2_test))
print('best_network_use:dayNet:{}  yearNet:{}'.format(best_flag_day, best_flag_year))
print(np.sum((test_true-test_predict) > 0))
print(np.sum((test_true-test_predict) < 0))
print(np.sum(test_true-test_predict))

if cut_co2 == False:
    data_co2_predict = np.load('co2_test_predict.npy')
    print(mk.original_test(test_predict-data_co2_predict))
if cut_co2:
    data_predict = np.load('test_predict.npy')
    print(mk.original_test(data_predict-test_predict))

utils.plot_data(test_predict-data_co2_predict)

l = len(test_predict)
plt.figure(figsize=(60, 10))
plt.plot(test_predict[0:l-2], c="orangered", label="test_predict", alpha=0.6)
plt.plot(test_true[0:l-2], c="darkblue", label="test_true", alpha=0.6)
plt.yticks(np.arange(int(min(test_predict)), int(max(test_predict) + 1), 1), fontsize=15)
# date = ['1956_01_01', '1969_08_10', '1983_03_19', '1996_10_28', '2010_06_07', '2023_12_31']
date = ['2017_3_13', '2018_07_24', '2019_12_05', '2021_04_16', '2022_08_23', '2023_12_31']
plt.xticks([0, int(0.2 * l), int(0.4 * l), int(0.6 * l), int(0.8 * l), l - 2], date, fontsize=25)
plt.xlabel("Date", fontsize=50)
plt.ylabel("t2m(°C)", fontsize=50)
plt.legend(fontsize=30)
plt.savefig('./result/combined_predict')
plt.show()


np.save('./result/MTTGnet_train_loss.npy', para_trainloss)
np.save('./result/MTTGnet_test_loss.npy', para_testloss)
np.save('./result/MTTGnet_r2_test.npy', para_r2_test)
np.save('./result/MTTGnet_rmse.npy', para_rmse)

# xepoch = np.linspace(1,1000,num=20)
# plt.figure()
# plt.plot(xepoch,para_trainloss[np.arange(0, 1000, 50)[:20]],"rx--",label = 'trainloss')
# plt.plot(xepoch,para_testloss[np.arange(0, 1000, 50)[:20]],"b+--",label = 'testloss')
# plt.legend()
# plt.ylabel('MSELoss')
# plt.xlabel('Number of epochs')
# plt.show()
#
# plt.figure()
# plt.plot(xepoch,para_r2_train[np.arange(0, 1000, 50)[:20]],"rx--",label = 'r2_train')
# plt.plot(xepoch,para_r2_test[np.arange(0, 1000, 50)[:20]],"b+--",label = 'r2_test')
# plt.legend()
# plt.ylabel('R2_score')
# plt.xlabel('Number of epochs')
# plt.show()

