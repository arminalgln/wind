# =============================================================================
# Liberaries
# =============================================================================
import pandas as pd
import os  # access files and so on
import sys  # for handling exceptions
import re  # for checking letter in a string
import numpy as np
import random
import time
import xlrd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import windforecast
from opencage.geocoder import OpenCageGeocode
import datetime
import math
import json
import urllib.request

class OpenWeatherAPI:
    def __init__(self, location_api, address, openweather_api):
        self.location_api = location_api
        self.address = address
        self.openweather_api = openweather_api


    def __get_address_lat_lng(self):
        geocoder = OpenCageGeocode(self.location_api)
        self.whole_location_info = geocoder.geocode(self.address)[0]
        geo = self.whole_location_info['geometry']
        lat, lng = geo['lat'], geo['lng']
        self.lat = lat
        self.lng = lng
# location_api_key='23e6edd3ccc7437b90c589fd7c9c6213'

    def get_forecasted_data(self):
        self.__get_address_lat_lng()
        base_url = 'https://api.openweathermap.org/data/2.5/onecall?lat=' + str(self.lat) + \
                   '&lon=' + str(self.lng) +'&%20exclude=hourly&appid=' + self.openweather_api

        with urllib.request.urlopen(base_url) as base_url:
            # print(search_address)
            data = json.loads(base_url.read().decode())


        hourly = data['hourly']
        hourly = pd.DataFrame(hourly)
        return hourly

class Ninja:
    def __init__(self, dst, train_ratio):
        self.dst = dst
        self.train_ratio = train_ratio


    def train_test_whole(self):
        historical = pd.read_csv(self.dst)
        normalized = historical[['electricity', 'wind_speed']].values
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized = min_max_scaler.fit_transform(normalized)
        df = pd.DataFrame(normalized, columns=['electricity_norm', 'wind_speed_norm'])
        historical[['electricity_norm', 'wind_speed_norm']] = df[['electricity_norm', 'wind_speed_norm']]

        data_size = historical.shape[0]
        resolution = 24 #hours in each sample
        total_samples = int(data_size/resolution)
        seperated_data = np.split(historical, total_samples, axis=0)
        random.seed(1)
        random.shuffle(seperated_data)
        train_number = int(self.train_ratio * total_samples)
        train = seperated_data[0:train_number]
        test = seperated_data[train_number:]

        x_train = []
        y_train = []
        for sample in train:
            x_train.append(list(sample['wind_speed_norm']))
            y_train.append(list(sample['electricity_norm']))
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = []
        y_test = []
        for sample in test:
            x_test.append(list(sample['wind_speed_norm']))
            y_test.append(list(sample['electricity_norm']))
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        return train, test, x_train, y_train, x_test, y_test

class SolcastHistorical:
    def __init__(self, dst):
        self.dst = dst
        self.data = self.__historical()

    def __time_add(self):
        historical = pd.read_csv(self.dst)
        ts = []
        for i in historical['PeriodEnd']:
            date = datetime.datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
            t = datetime.datetime.timestamp(date) - 7 * 3600  # Irvine to UTC difference
            ts.append(int(t))
        historical['t'] = ts
        return historical

    def __historical(self):

        historical = self.__time_add()

        return historical

class SolcastHistoricalTrainTest:
    def __init__(self, dst, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index
        self.dst = dst
        self.train,  self.test = self.__train_test_historical()

    def __time_add(self):
        historical = pd.read_csv(self.dst)
        ts = []
        for i in historical['PeriodEnd']:
            date = datetime.datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
            t = datetime.datetime.timestamp(date) - 7 * 3600  # Irvine to UTC difference
            ts.append(int(t))
        historical['t'] = ts
        return historical

    def __train_test_historical(self):

        historical = self.__time_add()

        train = {}
        count = 0
        for i, t in enumerate(self.train_index):
            start = t
            end = start + 24 * 3600
            part = historical.loc[(historical['t'] >= start) & (historical['t'] < end)]
            if part.shape[0] == 24:
                train[t] = part
                count += 1
        test = {}
        count = 0
        for i, t in enumerate(self.test_index):
            start = t
            end = start + 24 * 3600
            part = historical.loc[(historical['t'] >= start) & (historical['t'] < end)]
            if part.shape[0] == 24:
                test[t] = part
                count += 1

        return train, test

