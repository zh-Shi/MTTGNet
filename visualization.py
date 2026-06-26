import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os.path as osp
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import pandas as pd
import datetime
import utils

# dataset = netCDF4.Dataset('./data/ERA5_part/ERA5_temp/ERA5_part_2016_2023.nc', mode='r')
# time = dataset.variables['time'][:]
# lats = dataset.variables['latitude'][:]
# lons = dataset.variables['longitude'][:]
# t2m = dataset.variables['t2m'][:]
# # 将时间转换为可读格式（假设时间单位是自某个基准时间开始的天数）
# time_units = dataset.variables['time'].units
# dates = nc.num2date(time, time_units)
#
# # 绘制空间分布图（假设数据是3D的，维度是[time, lat, lon]，这里只绘制第一个时间步的数据）
# plt.figure(figsize=(12, 6))
# temperature_at_time0 = t2m[4320, :, :]  # 第一个时间步的数据
# plt.contourf(lons, lats, temperature_at_time0-273.15, cmap='coolwarm')
# plt.colorbar(label='Temperature(°C)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Temperature Distribution')
# plt.show()

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 二维图片绘制
# # 读取 NetCDF 文件
# file_path = './data/ERA5_part/ERA5_temp/ERA5_part_2016_2023.nc'
# dataset = nc.Dataset(file_path)
#
# # 打印文件信息，查看变量名称
# print(dataset)
#
# # 假设气温数据的变量名是 'temperature'，纬度、经度分别是 'lat'、'lon'
# temperature = dataset.variables['t2m'][:]
# lat = dataset.variables['latitude'][:]
# lon = dataset.variables['longitude'][:]
#
# # 提取第一个时间步的数据（假设数据是3D的，维度是[time, lat, lon]）
# temperature_at_time0 = temperature[480, :, :]
#
# # 绘制叠加在西安地图上的气温空间分布图
# fig = plt.figure(figsize=(12, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
#
# # 设置地图的范围为西安区域
# extent = [107., 110, 33, 35]  # [min_lon, max_lon, min_lat, max_lat]
# ax.set_extent(extent)
#
# # 添加地理特征
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)
#
# # 添加网格线
# gl = ax.gridlines(draw_labels=True)
# gl.top_labels = False
# gl.right_labels = False
#
# # 绘制气温数据
# contour = ax.contourf(lon, lat, temperature_at_time0-273.15, cmap='coolwarm', transform=ccrs.PlateCarree())
# plt.colorbar(contour, ax=ax, orientation='vertical', label='Temperature(°C)')
#
# # 添加西安市的标记（大致经纬度）
# xian_lon, xian_lat = 108.9398, 34.3416
# ax.plot(xian_lon, xian_lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
# ax.text(xian_lon + 0.1, xian_lat, 'Xi\'an', transform=ccrs.PlateCarree())
#
# plt.title('Temperature Distribution over Xi\'an')
# plt.show()

# y_pred = np.load("./result/ResGraphnet_test_predict.npy")
# y_true = np.load("./data/ERA5_part/ERA5_temp/era5_t2m_daily_1956_2023_mean_anomaly.npy")
# y_true = y_true[-y_pred.shape[0]:]
# # 创建一个新的 figure
# plt.figure(figsize=(8, 8))
# # 绘制散点图
# plt.scatter(y_true, y_pred, alpha=0.7, s=10)
# # 绘制完美预测的对角线
# min_val = min(min(y_true), min(y_pred))
# max_val = max(max(y_true), max(y_pred))
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
# # 添加标签和标题
# plt.xlabel('True Temperature(°C)', fontsize=15)
# plt.ylabel('Predicted Temperature(°C)', fontsize=15)
# plt.title('Linear Layer', fontsize=15)
# # 显示图形
# plt.show()

# x = np.load('./data/ERA5_part/ERA5_temp/era5_t2m_hourly_1956_2023_mean.npy')
# # 导入其他数据
# data_q = np.load('./data/ERA5_part/nontemp_data_process/era5_q_hourly_1956_2023_mean.npy')
# # data_tp = np.load('./data/ERA5_part/nontemp_data_process/era5_tp_hourly_1956_2023_mean.npy')
# # data_msl = np.load('./data/ERA5_part/nontemp_data_process/era5_msl_hourly_1956_2023_mean.npy')
# # data_tisr = np.load('./data/ERA5_part/nontemp_data_process/era5_tisr_hourly_1956_2023_mean.npy')
# # data_u10 = np.load('./data/ERA5_part/nontemp_data_process/era5_u10_hourly_1956_2023_mean.npy')
# data_v10 = np.load('./data/ERA5_part/nontemp_data_process/era5_v10_hourly_1956_2023_mean.npy')
# data = np.load('./data/ERA5_part/nontemp_data_process/data_mixed.npy')

def plot_figure_day(x):
    time = np.linspace(0, 802, 802)
    x_data_daily = []
    sum = 0
    flag = 0
    num = x[500168:517688].shape[0]
    for i in range(num):
        flag = flag + 1
        sum = sum + x[i]
        if flag == 24:
            x_data_daily.append(sum / 24)
            sum = 0
            flag = 0
    x_data_daily = np.array(x_data_daily)
    plt.figure()
    plt.plot(time[0:72], x[500096:500168], color="gray", linewidth=2.8)
    plt.plot(time[8:16], x[500104:500112], color="orange", linewidth=3, alpha=0.9)
    plt.plot(time[32:40], x[500128:500136], color="orange", linewidth=3, alpha=0.9)
    plt.axis('off')
    # # 隐藏刻度
    # plt.gca().axes.xaxis.set_ticklabels([])  # 隐藏横坐标刻度标签
    # plt.gca().axes.xaxis.set_ticks([])       # 隐藏横坐标刻度
    # plt.gca().axes.yaxis.set_ticklabels([])  # 隐藏纵坐标刻度标签
    # plt.gca().axes.yaxis.set_ticks([])       # 隐藏纵坐标刻度
    plt.axvline(x=24, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.axvline(x=48, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.axvline(x=72, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.show()

def plot_figure_year(x):
    time = np.linspace(0, 730, 730)
    x_data_daily = []
    sum = 0
    flag = 0
    num = x[500168:517688].shape[0]
    for i in range(num):
        flag = flag + 1
        sum = sum + x[i]
        if flag == 24:
            x_data_daily.append(sum / 24)
            sum = 0
            flag = 0
    x_data_daily = np.array(x_data_daily)
    plt.figure()
    plt.plot(time[0:730], x_data_daily[0:730], color="gray", linewidth=2.8)
    plt.plot(time[160:200], x_data_daily[160:200], color="b", linewidth=3, alpha=0.8)
    plt.plot(time[525:565], x_data_daily[525:565], color="b", linewidth=3, alpha=0.8)
    plt.axis('off')
    # # 隐藏刻度
    # plt.gca().axes.xaxis.set_ticklabels([])  # 隐藏横坐标刻度标签
    # plt.gca().axes.xaxis.set_ticks([])  # 隐藏横坐标刻度
    # plt.gca().axes.yaxis.set_ticklabels([])  # 隐藏纵坐标刻度标签
    # plt.gca().axes.yaxis.set_ticks([])  # 隐藏纵坐标刻度
    plt.axvline(x=365, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.show()

def plot_figure(x):
    time1 = np.linspace(0, 216, 72)
    time2 = np.linspace(216, 946, 730)
    x_data_daily = []
    sum = 0
    flag = 0
    num = x[500168:517688].shape[0]
    for i in range(num):
        flag = flag + 1
        sum = sum + x[i]
        if flag == 24:
            x_data_daily.append(sum / 24)
            sum = 0
            flag = 0
    x_data_daily = np.array(x_data_daily)
    plt.figure(figsize=(200, 8))
    plt.plot(time1[0:72], x[500096:500168]+0.0008, color="gray", linewidth=2.8)
    plt.plot(time1[8:16], x[500104:500112]+0.0008, color="orange", linewidth=3, alpha=0.9)
    plt.plot(time1[32:40], x[500128:500136]+0.0008, color="orange", linewidth=3, alpha=0.9)
    plt.plot(time2[0:730], x_data_daily[0:730], color="gray", linewidth=2.8)
    plt.plot(time2[160:200], x_data_daily[160:200], color="b", linewidth=3, alpha=0.8)
    plt.plot(time2[525:565], x_data_daily[525:565], color="b", linewidth=3, alpha=0.8)
    plt.axis('off')
    # # 隐藏刻度
    # plt.gca().axes.xaxis.set_ticklabels([])  # 隐藏横坐标刻度标签
    # plt.gca().axes.xaxis.set_ticks([])  # 隐藏横坐标刻度
    # plt.gca().axes.yaxis.set_ticklabels([])  # 隐藏纵坐标刻度标签
    # plt.gca().axes.yaxis.set_ticks([])  # 隐藏纵坐标刻度
    plt.axvline(x=0, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.axvline(x=72, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.axvline(x=216, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.axvline(x=581, color='k', linestyle='--', linewidth=3.5, alpha=0.7)
    plt.show()

# plot_figure(data_q)

# plot_figure_day(data_v10)
# plot_figure_year(data_v10)
# data_day = np.load("day_embedding.npy")
# data_year = np.load("year_embedding.npy")
# data = data_year[0:365, -1] + data_day[0:365, -1]
#
# plt.figure()
# plt.plot(data_day[370:735, -1], color="gray",linewidth=3.5)
# plt.axis('off')
# plt.show()
#
# plt.figure()
# plt.plot(data_year[0:365, -1], color="gray", linewidth=3.5)
# plt.axis('off')
# plt.show()

# data_prediction = np.load("test_predict.npy")
# plt.figure()
# plt.plot(data_prediction[0:365], color="gray", linewidth=3.5)
# plt.axis('off')
# plt.show()

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

# fig_si = (18, 10)
# fo_si = 15 # 图片名称
# fo_ti_si = 15  # 经纬度刻度
#
# lats = np.linspace(-90, 90, 721)
# lons = np.linspace(-180, 180, 1440)
#
# data = np.load('./data/ERA5_global/ERA5_total_cloud_cover_global_mean.npy')
# utils.plot_data(data)
# print(data)



# x_0 = np.load('./data/ERA5_global/ERA5_2m_temperature_global_mean_anomaly.npy')
# data = np.load('./data/ERA5_global/data_mixed_6hourly.npy')
#
# # 构造示例数据
# time = np.arange(0, 65, 1)
#
# v0 = x_0[1000:1065]
# v1 = data[0,1000:1065]
# v2 = data[3,1000:1065]
#
# # 定义历史和未来的分界点
# split = 60
#
# # 设置子图
# fig, axes = plt.subplots(3, 1, figsize=(4, 4), sharex=True)
#
# colors = ['green', 'blue', 'red']
# values = [v0, v1, v2]
# labels = [r"$V_0$", r"$V_1$", r"$V_2$"]
#
# for ax, val, color, label in zip(axes, values, colors, labels):
#     # 画历史部分
#     ax.plot(time[:split], val[:split], 'o-', color=color, alpha=0.8, label=label)
#     # 画未来部分（虚线）
#     ax.plot(time[split-1:], val[split-1:], 'o--', color=color, alpha=0.5)
#
#     # 添加图例
#     ax.legend(loc='upper left', fontsize=8, frameon=True)
#
#     # 加分界线
#     ax.axvline(split, color='k', linestyle='--')
#
# # 设置整体标题
# axes[0].set_title("history", fontsize=10, color="darkorange", loc="left")
# axes[0].set_title("future", fontsize=10, color="green", loc="right")
#
# # 设置横坐标
# axes[-1].set_xlabel("time")
#
# plt.tight_layout()
# plt.show()

x_0 = np.load('./data/ERA5_global/ERA5_t2m_daily_mean.npy')
start_date = '1980-01-01'
end_date = '2024-12-19'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
plt.title('Comparative Global Mean Temperature Predictions (January-March 2023)', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature Anomaly (°C)', fontsize=14) # 或 'Absolute Temperature (°C)'
plt.plot(date_range, x_0, label='Ground Truth', color='black', linestyle='-', marker='*', markersize=3, markevery=5)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.savefig('./fig/Fig4_Comparative_Predictions.png', dpi=300)
plt.show()
