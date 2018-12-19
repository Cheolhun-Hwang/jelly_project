import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('D:/data/csv/bu_jelly_modify.csv')
data_columns = list(dataset.columns.values)
X = dataset[dataset.columns[2:47]]
y = dataset[dataset.columns[52:]]

print(X.shape)
print(X)
print(y.shape)
print(y)

# fix random seed for reproducibility
np.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))

x_month = X.pop('month')
x_day = X.pop('day')
x_week = X.pop('week')
X.astype('float32')
y.astype('str')
X = pd.DataFrame(scaler.fit_transform(X), columns=data_columns[5:47])
X = pd.concat([x_month, x_week, X], axis=1)

data = pd.concat([X,y],axis=1)
print(data.head())

data.to_csv('D:/data/csv/bu_jelly_nom.csv', sep=',')


# plt.figure(figsize=(40, 40));
# plt.subplot(1,2,1);
# plt.plot(X['high_wind_speed'], color='red', label='high_wind_speed')
# plt.plot(X['avr_current_dir'], color='green', label='avr_current_dir')
# plt.plot(X['avr_current_speed'], color='blue', label='avr_current_speed')
# plt.plot(X['high_current_speed'], color='black', label='high_current_speed')
# plt.plot(X['avr_high_wave_height'], color='red', label='avr_high_wave_height')
# plt.plot(X['avr_air_temp'], color='green', label='avr_air_temp')
# plt.plot(X['avr_high_air_temp'], color='blue', label='avr_high_air_temp')
# plt.plot(X['high_air_temp'], color='black', label='high_air_temp')
# plt.plot(X['avr_air_press'], color='red', label='avr_air_press')
# plt.plot(X['avr_high_air_press'], color='green', label='avr_high_air_press')
# plt.plot(X['high_air_press'], color='blue', label='high_air_press')
# plt.plot(X['avr_water_temp'], color='black', label='avr_water_temp')
# plt.plot(X['avr_high_water_temp'], color='red', label='avr_high_water_temp')
# plt.plot(X['high_water_temp'], color='green', label='high_water_temp')
# plt.plot(X['avr_salinity'], color='blue', label='avr_salinity')
# plt.plot(X['avr_high_salinity'], color='black', label='avr_high_salinity')
# plt.plot(X['high_salinity'], color='red', label='high_salinity')
# plt.title('해파리 해양 변수 그래프')
# plt.xlabel('time [days]')
# plt.ylabel('normalize')
# plt.legend(loc='best')
# plt.show()

