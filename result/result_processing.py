import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk

# models = ['MTTGnet', 'ResGraphnet', 'GCN', 'GAT', 'UniMP', 'GIN', 'TAGCN']
#
# data_types = ['rmse', 'r2_test', 'train_loss', 'test_loss']
# data = {data_type: {model: np.load(f'./{model}_{data_type}.npy') for model in models} for data_type in data_types}
#
# data_predict = {model: np.load(f'./{model}_test_predict.npy') for model in models}
# data_true = np.load('era5_t2m_daily_1956_2023_mean_anomaly.npy')
# data_co2 = np.load('co2_test_predict.npy')
#
# day = 30
# x = np.linspace(0, day-1, day)
#
# epoch = 3000
# indices = np.arange(500, epoch, 50)
# # x = np.linspace(500, epoch, 50)
#
#
# plt.figure()
#
# styles = {
#     'MTTGnet': {'color': 'r', 'marker': 'x', 'label': 'MTTGNet', 'linestyle': '-'},
#     'ResGraphnet': {'color': 'b', 'marker': '+', 'label': 'ResGrpahNet', 'linestyle': '-'},
#     'GCN': {'color': 'limegreen', 'marker': '^', 'label': 'GCN', 'linestyle': '-'},
#     'GAT': {'color': 'gold', 'marker': 'o', 'label': 'GAT', 'linestyle': '-'},
#     'UniMP': {'color': 'purple', 'marker': 's', 'label': 'UniMP', 'linestyle': '-'},
#     'GIN': {'color': 'orange', 'marker': 'd', 'label': 'GIN', 'linestyle': '-'},
#     'TAGCN': {'color': 'cyan', 'marker': 'v', 'label': 'TAGCN', 'linestyle': '-'}
# }
#
# def plot_predict(data, ylabel, xlabel='Date'):
#     for model in models:
#         plt.plot(x, data[model][-365:day-365],
#                  color=styles[model]['color'],
#                  linestyle=styles[model]['linestyle'],
#                  marker=styles[model]['marker'],
#                  markersize=6,
#                  label=styles[model]['label'])
#     plt.plot(x, data_true[-365:day - 365], color='b', marker='h', label='True', linestyle='-')
#     plt.legend(loc='best')  # 固定图例位置
#     date = ['2023_01_01', '2023_01_07', '2023_01_13', '2023_01_19', '2023_01_25', '2023_01_31']
#     plt.xticks([0, int(0.2 * day), int(0.4 * day), int(0.6 * day), int(0.8 * day), day - 1], date, fontsize=15)
#     plt.ylabel(ylabel, fontsize=25)
#     plt.xlabel(xlabel, fontsize=25)
#     plt.show()
#     # plt.clf()
#     # for i in range(7):
#     #     plt.plot(x, data_true[-365:day-365], color='b', marker='h', label='True', linestyle='-')
#     #     plt.plot(x, data[models[i]][-365:day-365], color='r', marker='x', label=styles[models[i]]['label'], linestyle='-')
#     #     plt.plot()
#     #     plt.legend(loc='best')
#     #     date = ['2023_01_01', '2023_01_07', '2023_01_13', '2023_01_19', '2023_01_25', '2023_01_31']
#     #     plt.xticks([0, int(0.2 * day), int(0.4 * day), int(0.6 * day), int(0.8 * day), day - 1], date, fontsize=15)
#     #     plt.ylabel(ylabel, fontsize=15)
#     #     plt.xlabel(xlabel, fontsize=15)
#     # plt.show()
#
#
# def plot_metric(data, metric, ylabel, xlabel='Number of epochs'):
#     plt.clf()
#     for model in models:
#         plt.plot(x, data[metric][model][indices],
#                  color=styles[model]['color'],
#                  linestyle=styles[model]['linestyle'],
#                  marker=styles[model]['marker'],
#                  markersize=6,
#                  label=styles[model]['label'])
#     plt.legend(loc='best')
#     plt.ylabel(ylabel, fontsize=15)
#     plt.xlabel(xlabel, fontsize=15)
#     plt.show()
#
# # plot_metric(data, 'rmse', 'RMSE Loss')
# # plot_metric(data, 'r2_test', 'R2 Test')
# # plot_predict(data_predict, 't2m(°C)')
import utils

data_predict = np.load('../test_predict.npy')
data_predict_co2 = np.load('../co2_test_predict.npy')
data = data_predict - data_predict_co2
l = len(data)
plt.figure(figsize=(60, 10))
plt.plot(data, c="orangered", alpha=0.6)
plt.yticks(np.arange(int(min(data)), int(max(data) + 1), 0.5), fontsize=15)
date = ['2017_3_13', '2018_07_24', '2019_12_05', '2021_04_16', '2022_08_23', '2023_12_31']
plt.xticks([0, int(0.2 * l), int(0.4 * l), int(0.6 * l), int(0.8 * l), l - 2], date, fontsize=25)
plt.xlabel("Date", fontsize=27)
plt.ylabel("t2m(°C)", fontsize=27)
plt.show()
plt.savefig('../result/co2_change_predict')

utils.plot_data(data_predict_co2)


