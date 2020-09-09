import importlib
import windforecast
windforecast = importlib.reload(windforecast)
from windforecast import WindF
from windforecast import SolcastHistorical
from windforecast.readingdata import SolcastHistoricalTrainTest
from windforecast import OpenWeatherAPI
from windforecast import Ninja
import os  #access files and so on
import matplotlib
import matplotlib.pyplot as plt
# from solarforecast import SolarF
import numpy as np
import keras
import pandas as pd
import datetime
import time
from time import sleep
import schedule
import tensorflow as tf

#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#%%
####################################################
####################################################
####################################################
"""

wind forecasting

"""
####################################################
####################################################
####################################################
#%%
#get the forecasted data from openweather
address = True
address = '17 goodyear, Irvine'
if not address:
    address = input('where is your site address?  ')
location_api_key = '23e6edd3ccc7437b90c589fd7c9c6213'
openweather_API_key = 'b851dc4250e7c3b1c72f5ba2ec798741'
wf = OpenWeatherAPI(location_api_key, address, openweather_API_key)
pred = wf.get_forecasted_data()

#%%
windforecast = importlib.reload(windforecast)


#one year historical data from ninja
dst = 'data/ninja_etap.csv'
train_ratio = 0.7
data = Ninja(dst,train_ratio)
#%%
train, test, x_train, y_train, x_test, y_test = data.train_test_whole()
#%%
feature_numbers = 1
each_day_horizon = 24
window_horizon = 1

wind_forecaster = WindF(feature_numbers, (each_day_horizon * window_horizon), each_day_horizon)

wind_forecaster.opt_ls_mtr(optimizer='adam',
                                loss='mse',
                                metric='mse')
# #train

# y_train=y_train.reshape(327,48,1)
wind_forecaster.train(x_train, y_train, batch=10, epoch=100)
#evaluation on train set
# pmu_forecaster.solar_eval(x_train, y_train)
# #evaluation on dev set

# pmu_forecaster.solar_eval(x_train, y_train)
# pmu_forecaster.solar_eval(x_dev, y_dev)
# pmu_forecaster.solar_eval(x_test, y_test)

# pmu_forecaster.model.save('models/'+sc)


#%%
pred = wind_forecaster.wind_predict(x_train)
for i, k in enumerate(pred[0:10]):
    # print(i[30])
    # plt.plot(x_train[i])
    plt.plot(y_train[i])
    plt.plot(pred[i])
    plt.legend(['real','pred'])
    plt.show()


#%%



import pandas as pd
from sklearn import preprocessing


x = data[['electricity', 'wind_speed']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df=pd.DataFrame(x_scaled, columns = ['electricity', 'wind_speed'])
df[['time', 'local_time']] = data[['time', 'local_time']]

#%%
#hsitorical data from solcast

forecasted_features = ['Ghi', 'Ghi90', 'Ghi10', 'Ebh', 'Dni', 'Dni10', 'Dni90', 'Dhi',
       'air_temp', 'Zenith', 'Azimuth', 'cloud_opacity', 'period_end',
       'Period']
historical_features = ['PeriodEnd', 'PeriodStart', 'Period', 'AirTemp', 'AlbedoDaily',
       'Azimuth', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi',
       'PrecipitableWater', 'RelativeHumidity', 'SnowDepth', 'SurfacePressure',
       'WindDirection10m', 'WindSpeed10m', 'Zenith']

whole_training_features = ['Ghi', 'Ebh', 'Dni', 'Dhi', 'AirTemp', 'CloudOpacity'] #for today which will predict tomorrow
output_feature = ['PV_power']#for the next day power generation

%reload_ext windforecast

dst='data/solcast_etap_historical.csv'
hist = SolcastHistorical(dst)
start = 1546308000
end = 1577865600
data2019 = hist.data.loc[(hist.data['t'] >= start) & (hist.data['t'] < end)]

x = data2019[['WindSpeed10m']].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

data2019wind=pd.DataFrame(x_scaled, columns = ['WindSpeed10m'])
#%%
%matplotlib auto
a = 0
b = 8000
plt.plot(data2019['WindSpeed10m'].values[a:b])
plt.plot(data['wind_speed'].values[a:b])
plt.legend(['solcast 10 meter', 'ninja'])
plt.show()
#%%
%matplotlib auto
a = 0
b = 8000
plt.plot(data2019wind['WindSpeed10m'].values[a:b])
plt.plot(df['wind_speed'].values[a:b])
plt.legend(['solcast 10 meter', 'ninja'])
plt.show()


#%%
import scipy.stats as stats
r, p = stats.pearsonr(data2019['WindSpeed10m'].values[a:b], data['wind_speed'].values[a:b])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")


r, p = stats.pearsonr(data['electricity'].values[a:b], data['wind_speed'].values[a:b])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")

r, p = stats.pearsonr(data['electricity'].values[a:b], data2019['WindSpeed10m'].values[a:b])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")


#%%




#%%
def train_test_by_features(selected_features, hist, etap_power):

    train_features = []
    train_label = []
    for i in hist.train.keys():
        selected_data = hist.train[i][selected_features]
        train_features.append(selected_data.values)
        train_label.append(etap_power.train_data[i]['Avg'].values)

    train_features = np.array(train_features)
    train_label = np.array(train_label)

    test_features = []
    test_label = []
    for i in hist.test.keys():
        selected_data = hist.test[i][selected_features]
        test_features.append(selected_data.values)
        test_label.append(etap_power.test_data[i]['Avg'].values)

    test_features = np.array(test_features)
    test_label = np.array(test_label)
    #max min normalization
    powermax = np.max(train_label)
    powermin = np.min(train_label)

    feature_max = train_features.max(axis=(1,0))
    feature_min = train_features.min(axis=(1,0))

    ##normalize
    x_train = (train_features-feature_min)/(feature_max-feature_min)
    x_test = (test_features-feature_min)/(feature_max-feature_min)

    y_train = (train_label-powermin)/(powermax-powermin)
    y_test = (test_label-powermin)/(powermax-powermin)

    return x_train, x_test, y_train, y_test

# def normaly(x):
#     return [(i-min(x))/(max(x)-min(x)) for i in x]
#
# #%%
# samples=range(2)
# for sample in samples:
#     plt.plot((x_train[sample][:,0]))
#     plt.plot((y_train[sample]))
#     plt.legend(['f','p'])
#     plt.show(block=True)
#     plt.interactive(False)
# MSE_of_scenarios = {
#     'whole': 0.0012270294828340411,
#     'radiations': 0.018972916528582573,
#     'normal': 0.0009280733065679669,
#     'minimal': 0.0010660196421667933,
#     'Ghi': 0.0014592972584068775
#     }
#%%
# select features for training
scenarios = {
    'whole': ['Ghi', 'Ebh', 'Dni', 'Dhi', 'AirTemp', 'CloudOpacity'],
    'radiations':['Ghi', 'Ebh', 'Dni', 'Dhi'],
    'normal':['Ghi', 'AirTemp', 'CloudOpacity'],
    'minimal':['Ghi', 'CloudOpacity'],
    'Ghi':['Ghi']
}

for sc in scenarios:
    print(sc)
    selected_features = scenarios[sc]

    feature_numbers=len(selected_features)
    resolution=24
    x_train, x_test, y_train, y_test = train_test_by_features(selected_features, hist, etap_power)
    solar_forecaster = SolarF(feature_numbers,resolution)

    solar_forecaster.opt_ls_mtr(optimizer='adam',
                                loss='mse',
                                metric='mse')
# #train

    # y_train=y_train.reshape(327,48,1)
    solar_forecaster.train(x_train, y_train, batch=1, epoch=100)
    #evaluation on train set
    solar_forecaster.solar_eval(x_train, y_train)
    # #evaluation on dev set

    solar_forecaster.solar_eval(x_train, y_train)
    # solar_forecaster.solar_eval(x_dev, y_dev)
    solar_forecaster.solar_eval(x_test, y_test)

    solar_forecaster.model.save('models/'+sc)


#%%
x_train, x_test, y_train, y_test = train_test_by_features(selected_features, hist, etap_power)
solar_forecaster = SolarF(feature_numbers,resolution)

solar_forecaster.opt_ls_mtr(optimizer='adam',
                            loss='mse',
                            metric='mse')
# #train

# y_train=y_train.reshape(327,48,1)
solar_forecaster.train(x_train, y_train, batch=1, epoch=1)
#evaluation on train set
solar_forecaster.solar_eval(x_train, y_train)
# #evaluation on dev set

solar_forecaster.solar_eval(x_train, y_train)
# solar_forecaster.solar_eval(x_dev, y_dev)
solar_forecaster.solar_eval(x_test, y_test)

solar_forecaster.model.save('models/whole_features')

#%%
scenarios = {
    'whole': ['Ghi', 'Ebh', 'Dni', 'Dhi', 'AirTemp', 'CloudOpacity'],
    'radiations':['Ghi', 'Ebh', 'Dni', 'Dhi'],
    'normal':['Ghi', 'AirTemp', 'CloudOpacity'],
    'minimal':['Ghi', 'CloudOpacity'],
    'Ghi':['Ghi']
}

## loading model and compare their performance
mses={}
for sc in scenarios:
    if sc == 'normal':
        print(sc)
        selected_features = scenarios[sc]

        feature_numbers = len(selected_features)
        resolution = 24
        x_train, x_test, y_train, y_test = train_test_by_features(selected_features, hist, etap_power)

        loaded_model = keras.models.load_model('models/'+sc)
        print(loaded_model)
        predicted = loaded_model.predict(x_test)
        mse_error = loaded_model.evaluate(x_test, y_test)
        print(mse_error)
        mses[sc] = mse_error[0]
        # os.mkdir('models/figs/'+sc)
        # for i, k in enumerate(predicted):
        #     print(i)
        #     plt.plot(y_test[i])
        #     plt.plot(predicted[i])
        #     plt.legend(['real', 'pred'])
        #     plt.savefig('models/figs/' + sc + '/' + str(i) + '.png')
        #     plt.show()





#%%
#prediction
pred = solar_forecaster.solar_predict(x_test)
for i, k in enumerate(pred):
    # print(i[30])
    # plt.plot(x_train[i])
    plt.plot(y_test[i])
    plt.plot(pred[i])
    plt.legend(['real','pred'])
    plt.show()
# selected_data.head()

#%%
#saving keras model
solar_forecaster.model.save('models/whole_features')
#%%
loaded=keras.models.load_model('models/normal')





