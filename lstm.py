import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import matplotlib as plt

import numpy as np

df = pd.read_csv('ICAAP_daily_set.csv', sep=';',encoding='PT154')
dc=df.melt(id_vars=['date'])
dc['date']=pd.to_datetime(dc['variable'], format='%d.%m.%Y', errors='coerce')
del dc['variable']
dc['value'] = dc['value'].astype(str)
dc['value']=dc['value'].apply(lambda x:x.replace('%', ''))
dc['value'] = dc['value'].astype(float)
dc['date'] =pd.to_datetime(dc['date'])
#dc['mean']=dc.rolling(window='30').mean()
#print(dc)
#dc['date'] = dc['date'].dt.strftime('%Y-%m')
#dc['value'] = dc['value'].astype(float)
#avg=dc.groupby([dc['date']], sort=False).mean()
#avg2=dc.to_timestamp()

avg = dc.set_index('date').resample('M').mean()
print(avg)
#plt.plot(avg)
#plt.show()
#n=avg['date'].iloc[0]
#print(n)
def timeseries_to_supervised(data, lag=1):
    avg = pd.DataFrame(data)
    columns = [avg.shift(i) for i in range(1, lag + 1)]
    columns.append(avg)
    avg = pd.concat(columns, axis=1)
    avg.fillna(0, inplace=True)
    return avg


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    #test_scaled = scaler.transform(test)
    return scaler, train_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# load dataset
series = avg

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train = supervised_values#, supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled = scale(train)
'''''''''
df1 = pd.read_csv('ICAAP_daily BG2_dkbko_full2.csv', sep=';',encoding='PT154')
dc1=df1.melt(id_vars=['date'])
dc1['date']=pd.to_datetime(dc1['variable'], format='%d.%m.%Y', errors='coerce')
del dc1['variable']
dc1['value'] = dc1['value'].astype(str)
dc1['value']=dc1['value'].apply(lambda x:x.replace('%', ''))
dc1['value'] = dc1['value'].astype(float)
dc1['date'] =pd.to_datetime(dc1['date'], format='%Y-%m', errors='coerce')
dc1['date'] = dc1['date'].dt.strftime('%Y-%m')
dc1['value'] = dc1['value'].astype(float)
avg1=dc1.groupby([dc1['date']], sort=False).mean()
fact=avg1['value']
'''''''''
# repeat experiment
repeats = 5
error_scores = list()
df4=pd.DataFrame()
best_prediction = []
print(avg)
#best_proc = pd.DataFrame()
for r in range(repeats):
    # fit the model
    lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    forecast=lstm_model.predict(train_reshaped, batch_size=1)
    # walk-forward validation on the test data
    predictions = list()


    for i in range(11):
        # make one-step forecast
        X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat)
        # store forecast
        predictions.append(yhat[0])
    print(predictions)
    #print(type(predictions))
    if predictions > best_prediction:
       # best_model = model
        best_prediction = predictions
    print(best_prediction)


    # report performance
    #rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))

print(best_prediction)
'''''''''''''''
print('MSE test:'.format(
      mean_squared_error(fact, best_prediction)))

df1 = pd.read_csv('ICAAP_daily BG2_dkbko_full2.csv', sep=';',encoding='PT154')
dc1=df1.melt(id_vars=['date'])
dc1['date']=pd.to_datetime(dc1['variable'], format='%d.%m.%Y', errors='coerce')
del dc1['variable']
dc1['value'] = dc1['value'].astype(str)
dc1['value']=dc1['value'].apply(lambda x:x.replace('%', ''))
dc1['value'] = dc1['value'].astype(float)
dc1['date'] =pd.to_datetime(dc1['date'], format='%Y-%m', errors='coerce')
dc1['date'] = dc1['date'].dt.strftime('%Y-%m')
dc1['value'] = dc1['value'].astype(float)
avg1=dc1.groupby([dc1['date']], sort=False).mean()
'''''''''''''''
#avg['value'].plot()
#best_prediction.plot(color='r')

#plt.show()
#avg['date'] = pd.to_datetime(avg['date']).dt.normalize()
bp=pd.DataFrame(best_prediction)
avg2 = pd.concat([avg['value'], bp])

print(avg2)
#avg2['date'] = pd.to_datetime(avg2.index).dt.normalize()
print(avg2)

for n, i in enumerate(avg2.index):
    try:
        a=int(i)
        index=n-1
        break
    except:
        pass
n=len(avg)

#pd.date_range(start=avg2.index[index], periods=avg2.index.shape[0]-index, freq='M')[1:]
avg2.index=list(avg2.index[:index+1])+list(pd.date_range(start=avg2.index[index], periods=avg2.index.shape[0]-index, freq='M')[1:])
avg2=avg2.tail(n)
avg=np.round(avg,2)
avg2=np.round(avg2,2)

mini1=avg.min()
mini1=mini1.astype(str)
maxi1=avg.max()
maxi1=maxi1.astype(str)
mini2=avg2.min()
mini2=mini2.astype(str)
maxi2=avg2.max()
maxi2=maxi2.astype(str)
list_of_lists = [
  ['мин', mini1.values],
  ['макс', maxi1.values]]
#list_of_lists.to_string()
mini = pd.DataFrame(list_of_lists, columns=['история', 'значение'])
#mini.to_string()
list_of_lists1 = [
  ['мин', mini2.values],
  ['макс', maxi2.values]]

mini3 = pd.DataFrame(list_of_lists1, columns=['прогноз', 'значение'])
mini=mini.apply(lambda x:x.replace('[', ''))
mini=mini.apply(lambda x:x.replace(']', ''))
#mini['значения']=mini['значения'].apply(lambda x:x.replace(' ' ', ''))
#mini3=mini.apply(lambda x:x.replace('[', ''))
#mini3=mini.apply(lambda x:x.replace(']', ''))
avg.index = avg.index.map(lambda t: t.strftime('%Y-%m'))
avg2.index = avg2.index.map(lambda t: t.strftime('%Y-%m'))


print(mini)
#print(type(mini))
print(mini3)
#print(type(mini3))
# az.xlsx', 'w', encoding='PT154')
writer = pd.ExcelWriter('lstm.xlsx', engine='xlsxwriter')
#forecast.to_excel(writer, sheet_name='Sheet1')
#avg.to_excel(writer, sheet_name='Sheet2')
avg.to_excel(writer, sheet_name='Sheet1', startcol=0)
mini.to_excel(writer, sheet_name='Sheet1', startcol=3)
avg2.to_excel(writer, sheet_name='Sheet1', startcol=7)
mini3.to_excel(writer, sheet_name='Sheet1', startcol=10)
writer.save()
writer.close()
print(avg2)
#avg2.type()
#avg3 = pd.DataFrame(avg2, columns=['date', 'value'])

#print(avg3)
#avg3=pd.date_range(start=avg2.loc[np.where(avg2['row'] == 0)], periods=1, freq='M')
#print(avg3)
#avg2=avg2.to_timestamp()
#avg2 = avg2.resample('M')
"""""""""""""""
for(i in size:(size+12)){
  avg2$Date[i] <- avg2$Date[i+1]
}
"""""""""""""""


   # print('%d) Test RMSE: %.3f' % (r + 1, rmse))
    #error_scores.append(rmse)

# summarize results
#results = pd.DataFrame()
#results['rmse'] = error_scores
#print(results.describe())
#print(predictions, sep='\n')

"""""""""""""""
# plot residual errors
#residuals = pd.DataFrame(model_fit.resid)

#avg['forecast']=model_fit.predict(start=36,end=48,dynamic=True)
#avg[['value','forecast']].plot(figsize=(12,8))
#residuals.plot()
#plt.show()
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())

df1 = pd.read_csv('ICAAP_daily BG2_dkbko_full2.csv', sep=';',encoding='PT154')
dc1=df1.melt(id_vars=['date'])
dc1['date']=pd.to_datetime(dc1['variable'], format='%d.%m.%Y', errors='coerce')
del dc1['variable']
dc1['value'] = dc1['value'].astype(str)
dc1['value']=dc1['value'].apply(lambda x:x.replace('%', ''))
dc1['value'] = dc1['value'].astype(float)
dc1['date'] =pd.to_datetime(dc1['date'], format='%Y-%m', errors='coerce')
dc1['date'] = dc1['date'].dt.strftime('%Y-%m')
dc1['value'] = dc1['value'].astype(float)
avg1=dc1.groupby([dc1['date']], sort=False).mean()
print(avg1)
"""""""""""""""
