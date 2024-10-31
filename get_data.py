import cdsapi
import calendar
import os
c = cdsapi.Client()

dic ={
    'product_type': 'reanalysis', #选择数据集
    'format': 'netcdf',  #选择数据格式
    'variable':['10m_u_component_of_wind', '10m_v_component_of_wind',
                'mean_sea_level_pressure', 'toa_incident_solar_radiation',
                'total_precipitation',], #选择要素
    'variable': 'specific_humidity',
    'year': [
            '1956', '1957', '1958',
            '1959', '1960', '1961',
            '1962', '1963', '1964',
            '1965', '1966', '1967',
            '1968', '1969', '1970',
            '1971', '1972', '1973',
            '1974', '1975', '1976',
            '1977', '1978', '1979',
            '1980', '1981', '1982',
            '1983', '1984', '1985',
            '1986', '1987', '1988',
            '1989', '1990', '1991',
            '1992', '1993', '1994',
            '1995', '1996', '1997',
            '1998', '1999', '2000',
            '2001', '2002', '2003',
            '2004', '2005', '2006',
            '2007', '2008', '2009',
            '2010', '2011', '2012',
            '2013', '2014', '2015',
            '2016', '2017', '2018',
            '2019', '2020', '2021',
            '2022', '2023',
        ],
    'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
    'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
    'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            35, 107, 33,
            110,
        ],
        'pressure_level': '1000',
}
# for i in range(1956, 2024):
#     for j in range(1, 13):
#         day_num = calendar.monthrange(i, j)[1]  # 根据年月，获取当月日数
#         dic['year'] = str(i)
#         dic['month'] = str(j).zfill(2)
#         dic['day'] = [str(d).zfill(2) for d in range(1, day_num + 1)]
#         # filename = 'D:\\Python\\PyCharm\\PytorchProject\\GaTun-main\\data_temp\\data\\ERA5_monthly' + '_' + str(i) + '_' + str(j).zfill(2)  + '.nc'  # 文件存储路径
#         filename = 'D:\Python\PyCharm\TTGNet\ERA5_part\ERA5_nontemp\ERA5_part' + '_' + str(i) + '_' + str(j).zfill(2)  + '.nc'  # 文件存储路径
#         if os.path.isfile(filename):
#             # print(f"{filename} 已存在，跳过下载")
#             continue

# c.retrieve('reanalysis-era5-pressure-levels', dic, 'D:\Python\PyCharm\TTGNet\ERA5_part\ERA5_humidity\ERA5_part_1956_2023.nc')  #下载数据

for i in range(1956, 2024):
    dic['year'] = str(i)
    filename = 'D:\Python\PyCharm\TTGNet\ERA5_part\ERA5_humidity\ERA5_part' + '_' + str(i) + '.nc'  # 文件存储路径
    if os.path.isfile(filename):
        # print(f"{filename} 已存在，跳过下载")
        continue
    c.retrieve('reanalysis-era5-pressure-levels', dic, filename)  # 下载数据
