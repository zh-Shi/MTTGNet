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

#固定随机数
def seed_everything(seed):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(100)

# #导入数据
# data = np.load('./ERA5_part/era5_t2m_hourly_1956_2023.npy')
# data_part = np.load('./ERA5_part/era5_t2m_hourly_1956_2023_one.npy')

#设置超参数
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
l_x = 60
l_y = 1
hidden_dim = 64
lr = 0.0001
weight_decay = 5e-4
epochs = 1000
ratio_train = 0.6
num_layers = 1
cut_co2 = False

model_name_all = ["ResGraphnet", "GCN", "Cheb", "GIN", "UniMP", "TAGCN", "GAT", "LSTM", "GRU", "MTTGNet"]
model_name = 'ResGraphnet'

# x = np.load('./data/ERA5_part/ERA5_temp/era5_t2m_daily_1956_2023_mean_anomaly.npy')
# num = x.shape[0]
#
# # 导入其他数据
# data = np.load('./data/ERA5_part/nontemp_data_process/data_mixed.npy')
# x = np.vstack((x, data))
#
# num_train = int(ratio_train * num)
# if x.ndim == 2:
#     data_train, data_test = x[:, :num_train], x[:, num_train:num]
# if x.ndim == 1:
#     data_train, data_test = x[:num_train], x[num_train:num]
#
# # 削减co2浓度
# if cut_co2:
#     x[7, :] = np.concatenate((x[7, :num_train-1000], 2*x[7, num_train-1000]-x[7, num_train-1000:]))

x = pd.read_csv("./data/data_all.csv")
x = x.iloc[100000:120000]
x = x[['p10', 'kp', 'bulkspeed']].values
num = x.shape[0]
x = x.transpose()
num_train = int(ratio_train * num)
if x.ndim == 2:
    data_train, data_test = x[:, :num_train], x[:, num_train:num]
if x.ndim == 1:
    data_train, data_test = x[:num_train], x[num_train:num]

start_time = time.time()
x_train, y_train = utils.create_inout_sequences(data_train, l_x, l_y)
x_test, y_test = utils.create_inout_sequences(data_test, l_x, l_y)
print(time.time()-start_time)


x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

num_nodes = x_train.shape[0] + x_test.shape[0]

x = torch.cat((x_train, x_test), dim=0)
y = torch.cat((y_train, y_test), dim=0)

adm_day = utils.path_graph(num_nodes,'day')

edge_index, edge_weight = utils.tran_adm_to_edge_index(adm_day)

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


model = net.Contrast_GNNmodel(l_x, hidden_dim, l_y, edge_weight, model_name).to(device)
# model = net.RNNTime(model_name, 1, hidden_dim, l_y, l_x, num_layers).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
edge_index = edge_index.to(device)

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
    output = model(x, edge_index)
    # output = (output + 1) / 2 * mm1 + mn1
    output_train, y_train = output[train_mask], y[train_mask]
    train_loss = criterion(output_train[:, -1], y_train[:, -1])
    para_trainloss[epoch] = train_loss
    train_loss.backward()
    optimizer.step()

    model.eval()
    output_test, y_test = output[test_mask], y[test_mask]
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
        if cut_co2:
            np.save('./result_new/{}_co2_test_predict.npy'.format(model_name), test_predict)
        if cut_co2 == False:
            np.save('./result_new/{}_test_predict.npy'.format(model_name), test_predict)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))
        mse = np.mean(np.square(train_predict - train_true))
        print("mse: {:.8f}".format(mse))



end_time = time.time()
run_time = end_time - start_time
print("Run_time:{}".format(run_time))
print('best model epoch:{}'.format(ep))

print("Best:RMSE_Train={:.5f}  RMSE_Test={:.5f}  R2_Train={:.7f}  R2_Test={:.7f}".
      format(best_rmse_train, best_rmse_test, best_r2_train, best_r2_test))




l = len(test_predict)
plt.figure(figsize=(60, 10))
plt.plot(test_predict[0:l-2], c="orangered", label="test_predict", alpha=0.6)
plt.plot(test_true[0:l-2], c="darkblue", label="test_true", alpha=0.6)
plt.yticks(np.arange(int(min(test_predict)), int(max(test_predict) + 1), 0.5))
# date = ['1956_01_01', '1969_08_10', '1983_03_19', '1996_10_28', '2010_06_07', '2023_12_31']
# date = ['2017_3_13', '2018_07_24', '2019_12_05', '2021_04_16', '2022_08_23', '2023_12_31']
# plt.xticks([0, int(0.2 * l), int(0.4 * l), int(0.6 * l), int(0.8 * l), l - 2], date)
plt.xlabel("Date", fontsize=50)
plt.ylabel("t2m", fontsize=50)
plt.legend(fontsize=30)
plt.show()


np.save('./result_new/{}_r2_test.npy'.format(model_name), para_r2_test)
np.save('./result_new/{}_rmse.npy'.format(model_name), para_rmse)

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

