#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:51:14 2021

@author: ppp
"""

import requests
import time
import os
import pmdarima as pm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import pandas as pd
import numpy as np
import torch



def encode_query(queryDate=None, queryHour=None):
    if queryDate is None:
        #queryTimestamp = pd.Timestamp.now()
        queryTimestamp = pd.Timestamp.now(tz=os.environ['TZ_DJANGO'])
        queryTimestamp = queryTimestamp.tz_localize(None)
    else:
        queryDateTime = queryDate +' '+ queryHour
        queryTimestamp = pd.Timestamp(queryDateTime,freq='h')
    return queryTimestamp



def get_timestamps_training_data(queryTimestamp, timespan=24*7*4*2, unit='h'):
    #nowTimestamp = pd.Timestamp.now().round('h')
    nowTimestamp = pd.Timestamp.now(tz=os.environ['TZ_DJANGO']).round('h')
    nowTimestamp = nowTimestamp.tz_localize(None)
    if (queryTimestamp < nowTimestamp):
        startTimestamp=queryTimestamp-pd.to_timedelta(timespan,unit=unit)
        endTimestamp=queryTimestamp
    else:
        startTimestamp=nowTimestamp-pd.to_timedelta(timespan,unit=unit)
        endTimestamp=nowTimestamp
    startTimestamp = startTimestamp.tz_localize(None)
    endTimestamp = endTimestamp.tz_localize(None)
    return startTimestamp, endTimestamp

def get_spot_market_training_data(startTimestamp, endTimestamp, tokenfile="ami/spotcasting/montel_bearer.json"):
    r = requests.get('https://coop.eikon.tum.de/mbt/mbt.json')
    token_data = r.json()
    startDateTime=startTimestamp.strftime('%Y-%m-%d')
    endDateTime=endTimestamp.strftime('%Y-%m-%d')

    url = 'http://api.montelnews.com/spot/getprices'
    headers = {"Authorization": "Bearer " + token_data['access_token'] }
    params = {'spotKey': '14',
            'fields': ['Base', 'Peak','Hours'],
            'fromDate': startDateTime ,
            'toDate': endDateTime,
            'currency': 'eur',
            'sortType': 'Ascending'}
    response = requests.get(url, headers=headers, params=params)

    sp = pd.DataFrame(pd.json_normalize(response.json()['Elements'],record_path='TimeSpans', meta='Date'))
    sp['DateTime'] = pd.to_datetime(sp['Date'].str[:10] + ' ' + sp['TimeSpan'].str[:5])
    dti = pd.DatetimeIndex(sp['DateTime'], freq='h')
    #dti = dti.tz_localize(tz=os.environ['TZ_DJANGO'])
    sp.set_index(dti, inplace=True)
    return sp[startTimestamp:endTimestamp]


def get_top_feature_names():
    feature_names = [
            'value__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
            'value__root_mean_square',
            'value__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0',
            'value__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
            'value__mean_second_derivative_central',
            'value__fft_coefficient__attr_"imag"__coeff_3',
            'value__cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)',
            'value__quantile__q_0.8', 'value__index_mass_quantile__q_0.9',
            'value__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
            'value__mean_change', 'value__fft_coefficient__attr_"real"__coeff_8',
            'value__fft_coefficient__attr_"real"__coeff_9',
            'value__fft_aggregated__aggtype_"centroid"',
            'value__fft_coefficient__attr_"real"__coeff_10',
            'value__fft_coefficient__attr_"real"__coeff_12',
            'value__fft_coefficient__attr_"real"__coeff_11'
        ]
    return feature_names
def get_exogenous_features(queryTimestamp, prediction_length, features):
    # get current time
    #nowTimestamp = pd.Timestamp.now().round('h')
    nowTimestamp = pd.Timestamp.now(tz=os.environ['TZ_DJANGO']).round('h')
    nowTimestamp = nowTimestamp.tz_localize(None)
    # get date indeces
    train_start, train_end = get_timestamps_training_data(queryTimestamp)
    test_end = train_end + pd.to_timedelta(prediction_length-1,unit='h')

    # get label
    label = get_spot_market_training_data(train_start, test_end)

    label = pd.DataFrame({'value': label.Value, 'date': label.index}).set_index('date')

    test_non_end = train_end + pd.to_timedelta(prediction_length, unit='h')
    test_non_label = label[train_end-pd.to_timedelta('1h'):test_non_end]

    label = label.fillna(method='bfill')
    # split
    train_label = label[train_start:train_end]
    test_label = label[train_end:test_end]

    if len(test_label) < prediction_length:
        new_test_label = pd.DataFrame(
            {'value': np.nan, 'date': pd.date_range(start=train_end, end=test_end, freq='h')}).set_index('date')
        new_test_label.loc[test_label.index] = test_label
        test_label = new_test_label

    if queryTimestamp < datetime.datetime(2021,2,27):
        train_features = []
        test_features = []
    else:
        # get mask
        reference_label = pd.read_pickle('ami/spotcasting/label.pkl')
        date_range = pd.date_range(start=train_start,end=test_end,freq='h')
        normal_mask = (date_range >= reference_label.index[0]) & (date_range <= reference_label.index[-1]) # label + features
        match_feature_mask = (date_range < nowTimestamp) & ((date_range < reference_label.index[0])|(date_range > reference_label.index[-1]))
        predict_feature_mask = date_range >= nowTimestamp

        # get the features
        feature_names = get_top_feature_names()
        # normal (we have corresponding features in cache)
        total_features = pd.DataFrame(columns=feature_names,index=date_range)
        total_features[normal_mask] = np.array(features.loc[date_range[normal_mask]])

        # match features (we have label data, we can find approximate feature part using spot price)
        reference_label = reference_label.loc['2021-05-01 00:00:00':]
        target_label = label.loc[date_range[match_feature_mask]]
        best_mae = float('inf')
        best_index = 0
        for index in reference_label.index:
            ref_label = reference_label[index:index+pd.Timedelta(len(target_label)-1, unit='h')]
            if len(ref_label) == len(target_label):
                mae = sum(abs(np.array(ref_label.value) - np.array(target_label.value)))
                if mae < best_mae:
                    best_mae = mae
                    best_index = index
            else:
                break
        total_features[match_feature_mask] = np.array(features[best_index:best_index+pd.Timedelta(len(target_label)-1, unit='h')])

        # predict features (we do not have label and feature, we need to predict features)
        total_features[predict_feature_mask] = \
            np.array(total_features[nowTimestamp-pd.Timedelta(predict_feature_mask.sum(), unit='h'):nowTimestamp-pd.Timedelta('1h')])

        # split
        train_features = total_features[train_start:train_end]
        test_features = total_features[train_end:test_end]

    return train_label, test_label, train_features, test_features, test_end, test_non_label

def AMI_ARIMA(train_label, test_label, train_features, test_features):
    if train_label.index[-1] <datetime.datetime(2021,2,27):
        predict_values = np.array(test_label)
    else:
        model = pm.ARIMA(order=(2,1,2))
        train_X = StandardScaler().fit_transform(np.array(train_features))
        test_X = StandardScaler().fit_transform(np.array(test_features))
        model.fit(train_label, X=train_X)
        predict_values = model.predict(n_periods=len(test_X), X=test_X)
    
    predict_values = pd.DataFrame(predict_values, index=test_label.index)
    return predict_values

def generate_positional_coding(startTimestamp, endTimestamp):
    month_list = []
    day_list = []
    hour_list = []
    startTimestamp_list = []

    while startTimestamp <= endTimestamp:
        month_list.append(startTimestamp.month)
        day_list.append(startTimestamp.day)
        hour_list.append(startTimestamp.hour)
        startTimestamp_list.append(startTimestamp)
        startTimestamp += datetime.timedelta(hours=1)

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    df = pd.DataFrame()
    df['sin_hour'] = np.sin(2 * np.pi * np.array(hour_list) / hours_in_day)
    df['cos_hour'] = np.cos(2 * np.pi * np.array(hour_list) / hours_in_day)
    df['sin_day'] = np.sin(2 * np.pi * np.array(day_list) / days_in_month)
    df['cos_day'] = np.cos(2 * np.pi * np.array(day_list) / days_in_month)
    df['sin_month'] = np.sin(2 * np.pi * np.array(month_list) / month_in_year)
    df['cos_month'] = np.cos(2 * np.pi * np.array(month_list) / month_in_year)
    df.index = startTimestamp_list
    return df


def transformer_predict(querydate, queryHour, prediction_length, model=None):
    # get date indices
    #timespan = {1: 72, 24: 72, 24 * 7: 24 * 7 * 2}
    timespan = {1+1: 6, 24+1: 18, 24 * 7+1: 24 * 7}
    assert (prediction_length in timespan.keys()), 'prediction_length can only be 1, 24, 24*7'
    queryTimestamp = encode_query(querydate, queryHour)
    input_start, input_end = get_timestamps_training_data(queryTimestamp, timespan[prediction_length])
    input_start_new, _ = get_timestamps_training_data(queryTimestamp, 400)
    
    test_end = input_end + pd.to_timedelta(prediction_length - 1, unit='h')
    input_end1 = input_end - pd.to_timedelta(1, unit='h')


    # get data & generate positional coding
    data1 = get_spot_market_training_data(input_start_new, test_end)
    pos_df = generate_positional_coding(input_start, test_end)
    data = get_spot_market_training_data(input_start, test_end)

    data = pd.DataFrame({'value': data.Value, 'date': data.index}).set_index('date')
    data = data.fillna(method='bfill')
    data = pd.concat([data, pos_df], axis=1, join='inner')
    

    pos_df1 = generate_positional_coding(input_start_new, test_end)
    data1 = pd.DataFrame({'value': data1.Value, 'date': data1.index}).set_index('date')
    test_non_end = input_end + pd.to_timedelta(prediction_length, unit='h')
    test_non_label = data1[input_end1:test_non_end]
    data1 = data1.fillna(method='bfill')
    output_train_label = data1[input_start_new:input_end]
    data1 = pd.concat([data1, pos_df1], axis=1, join='inner')

    input = data1[input_start:input_end1]

    # normalize & to tensor
    scaler = MinMaxScaler()
    input['value'] = scaler.fit_transform(input['value'].values.reshape(-1, 1))
    input = torch.tensor(input.values).unsqueeze(1).float()
    pos = torch.tensor(pos_df.values).unsqueeze(1).float()
    next_input_model = input

    # predict
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for i in range(prediction_length):
            prediction = model(next_input_model, device='cpu')
            if all_predictions == []:
                all_predictions = prediction[-1, :, :].unsqueeze(0)
            else:
                all_predictions = torch.cat((all_predictions, prediction[-1, :, :].unsqueeze(0)))
            if i != prediction_length-1:
                pos_encoding_old_vals = input[1:, :, 1:] if i == 0 else pos_encodings[1:, :, :]
                pos_encoding_new_val = pos[timespan[prediction_length]+i, :, :].unsqueeze(0)
                pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val))

                next_input_model = torch.cat((next_input_model[1:, :, 0].unsqueeze(-1), prediction[-1, :, :].unsqueeze(0)))
                next_input_model = torch.cat((next_input_model, pos_encodings), dim=2)

    # to numpy & denormalize
    all_predictions = all_predictions.numpy()
    all_predictions = all_predictions.reshape(-1, 1)
    all_predictions = scaler.inverse_transform(all_predictions)

    pred_df = pd.DataFrame(data=all_predictions, index=pos_df.index[timespan[prediction_length]:], columns=['value'])
    # return pred_df.values, test_label['value'].values
    return output_train_label, pred_df, test_end, test_non_label



