import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os.path as osp
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap

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