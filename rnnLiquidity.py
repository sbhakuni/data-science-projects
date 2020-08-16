# -*- coding: utf-8 -*-
"""
Created on Tue Jul  19 18:29:25 2019

@author: bhaku
"""
import pandas as pd
import numpy as np
from AllFunctionsLiq  import *

"""in this code i have used recurrent neural networks to predict the trends of various liquidity measures(one at a time)"""

##############################################################################
clean_data=CleanTRTHdata(data)# was done on the previous large consolidated excel file
agnc=TradingHours(clean_data['AGNC.OQ'])
agnc30sec=Aggregation(agnc,30)
depth=ApplyMeasure(agnc30sec,  Depth, series_name = 'measure value')
depth_df=pd.Series.to_frame(depth)


training_set_depth = depth_df.iloc[:3000, 0:1].values
test_set_depth=depth_df.iloc[3000:3120, 0:1].values 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set_depth)

X_train = []
y_train = []
for i in range(120, 3000):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 25, batch_size = 32)


dataset_total = pd.concat((pd.DataFrame(training_set_depth), pd.DataFrame(test_set_depth)), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_set_depth) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 240):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_depth= regressor.predict(X_test)
predicted_depth = sc.inverse_transform(predicted_depth)

# Visualising the results
import matplotlib.pyplot as plt
plt.plot(test_set_depth, color = 'red', label = 'test_set_depth')
plt.plot(predicted_volume, color = 'blue', label = 'predicted depth')
plt.title('depth Prediction')
plt.xlabel('Time')
plt.ylabel('depth')
plt.legend()
plt.show()
plt.plot(test_set_volume, color = 'red', label = 'test_set_depth')
plt.show()
plt.plot(predicted_volume, color = 'blue', label = 'predicted depth')
plt.show()

####################################################################################
clean_data=CleanTRTHdata(data)
agnc=TradingHours(clean_data['AGNC.OQ'])
agnc30sec=Aggregation(agnc,30)
logRelativeSpreadLogPrice=ApplyMeasure(agnc30sec,  LogRelativeSpreadLogPrice, series_name = 'measure value')
logRelativeSpreadLogPrice_df=pd.Series.to_frame(logRelativeSpreadLogPrice)
logRelativeSpreadLogPrice_df=logRelativeSpreadLogPrice_df.fillna(method='ffill')

#a=pd.DataFrame(agnc30sec.items())
#logRelativeSpreadLogPrice_df=pd.DataFrame(logRelativeSpreadLogPrice)
training_set_logRelativeSpreadLogPrice = logRelativeSpreadLogPrice_df.iloc[:3000, 0:1].values
test_set_logRelativeSpreadLogPrice=logRelativeSpreadLogPrice_df.iloc[3000:3120, 0:1].values 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set_logRelativeSpreadLogPrice)

X_train = []
y_train = []
for i in range(120, 3000):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 200, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 200, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 200))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 20, batch_size = 32)


dataset_total = pd.concat((pd.DataFrame(training_set_logRelativeSpreadLogPrice), pd.DataFrame(test_set_logRelativeSpreadLogPrice)), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_set_logRelativeSpreadLogPrice) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 240):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_logRelativeSpreadLogPrice= regressor.predict(X_test)
predicted_logRelativeSpreadLogPrice = sc.inverse_transform(predicted_logRelativeSpreadLogPrice)

# Visualising the results
import matplotlib.pyplot as plt
plt.plot(test_set_logRelativeSpreadLogPrice, color = 'red', label = 'test_set_OrderRatio')
plt.plot(predicted_logRelativeSpreadLogPrice, color = 'blue', label = 'predicted OrderRatio')
plt.title('logRelativeSpreadLogPrice Prediction')
plt.xlabel('Time')
plt.ylabel('logRelativeSpreadLogPrice')
plt.legend()
plt.show()
plt.plot(test_set_logRelativeSpreadLogPrice, color = 'red', label = 'test_set_OrderRatio')
plt.show()
plt.plot(predicted_logRelativeSpreadLogPrice, color = 'blue', label = 'predicted OrderRatio')
plt.show()
############################################################################
