import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 用于更美观的绘图和可能的平滑曲线

# --- 1. 加载数据和日期范围 ---
x_0 = np.load('./data/ERA5_global/ERA5_t2m_daily_mean.npy')

# 检查数据长度和日期范围是否匹配
# 假设x_0是从1980-01-01到2024-12-19的每日数据
start_date_data = pd.to_datetime('1980-01-01')
end_date_data = pd.to_datetime('2024-12-19')
date_range_data = pd.date_range(start=start_date_data, end=end_date_data, freq='D')

# 确保数据长度一致，防止索引错误
if len(x_0) != len(date_range_data):
    print(f"Warning: Data length ({len(x_0)}) does not match date range length ({len(date_range_data)}). Adjusting date range.")
    date_range_data = date_range_data[:len(x_0)] # 截断日期范围以匹配数据长度
    # 或者，如果数据太短，可能需要加载更多数据或调整起始日期

# 将数据转换为 Pandas Series，方便处理时间序列
t2m_series = pd.Series(x_0, index=date_range_data)

# --- 2. 计算温度异常 ---
# 选择一个参考期，例如 1981-2010 年作为基准期
# 注意：你的数据从 1980 年开始，如果选择 1950-1980 就不完整，
# 所以选择一个在你数据范围内的完整时期作为参考基准更为合理。
reference_start = '1980-01-01'
reference_end = '2024-12-31'

# 提取参考期内的温度数据
reference_period_data = t2m_series[reference_start:reference_end]
if reference_period_data.empty:
    raise ValueError(f"Reference period data is empty. Check dates {reference_start} to {reference_end} against your dataset range {date_range_data.min()} to {date_range_data.max()}.")

# 计算参考期的平均温度
mean_t2m_reference = reference_period_data.mean()
print(f"Reference period ({reference_start} to {reference_end}) mean temperature: {mean_t2m_reference:.2f}°C")

# 计算所有时间步的温度异常
temperature_anomaly = t2m_series - mean_t2m_reference

# --- 3. 平滑趋势线 (可选，但强烈推荐用于凸显趋势) ---
# 使用滚动平均来平滑日变化，显示长期趋势
# 例如，365天的滚动平均可以平滑掉季节性变化和大部分日变化，直接显示年际趋势
rolling_mean_window = 365 # 365天滚动平均
smoothed_anomaly = temperature_anomaly.rolling(window=rolling_mean_window, center=True).mean()

# --- 4. 绘图 ---
plt.figure(figsize=(15, 8))

# 绘制每日温度异常
plt.plot(temperature_anomaly.index, temperature_anomaly, label='Daily Temperature Anomaly', color='skyblue', alpha=0.6, linewidth=0.8)

# 绘制平滑后的趋势线，用于凸显上升趋势
plt.plot(smoothed_anomaly.index, smoothed_anomaly, label=f'{rolling_mean_window}-Day Rolling Mean Anomaly', color='red', linewidth=2.5)

# 绘制零线，表示与参考期平均值持平
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label=f'{reference_start}-{reference_end} Mean')

# 标题和标签
plt.title(f'Global Mean Temperature Anomaly ({start_date_data.year}-{end_date_data.year})', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=20)
plt.ylabel(f'Temperature Anomaly (°C) relative to {reference_start}-{reference_end}', fontsize=16)

# 图例和网格
plt.legend(fontsize=15, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# 优化 X 轴刻度，只显示年份
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5)) # 每5年一个刻度
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout() # 自动调整布局，防止标签重叠
plt.savefig('./fig/Global_Mean_Temperature_Anomaly.png', dpi=300)
plt.show()