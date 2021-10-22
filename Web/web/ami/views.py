from django.shortcuts import render
#from .utils import get_plot
from django.utils import timezone
from .spotcasting import basicforecast as bf
#from sktime.forecasting.naive import NaiveForecaster
import numpy as np
import pandas as pd
from datetime import datetime
from .model import Transformer
import torch
import os

def load_model(path):
    model = Transformer(dropout=0.1)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

# Create your views here.
def test(request):

    nowTimestamp = pd.Timestamp.now(tz=os.environ['TZ_DJANGO'])
    year = nowTimestamp.year
    month = '{:02d}'.format(nowTimestamp.month)
    day = '{:02d}'.format(nowTimestamp.day)

    date_limit = str(year) + "-" + str(month) + "-" + str(day)
    hour_limit = str(nowTimestamp.round('h').hour) + ':' + "00"


    context = {
        "date_limit": date_limit,
        "hour_limit" : hour_limit

    }

    return render(request, "ami/index.html", context)

chart_arima = False
chart_trf = False

data_arima = None
time_arima = None
train_arima = None
timesteps_arima = None
pred_arima = None
plot_arima = None
predict_range_arima = None


data_trf = None
time_trf = None
train_trf = None
timesteps_trf = None
pred_trf = None
plot_trf = None
predict_range_trf = None

features = pd.read_pickle('ami/2021_features.pkl')
features = features[bf.get_top_feature_names()]
model = load_model('ami/save_model/best_train_182.pth')

def c_test(request):
    global chart_arima


    global chart_trf

    global data_arima
    global time_arima
    global train_arima
    global timesteps_arima
    global pred_arima
    global plot_arima
    global predict_range_arima

    global data_trf
    global time_trf
    global train_trf
    global timesteps_trf
    global pred_trf
    global plot_trf
    global predict_range_trf


    nowTimestamp = pd.Timestamp.now(tz=os.environ['TZ_DJANGO'])
    year = nowTimestamp.year
    month = '{:02d}'.format(nowTimestamp.month)
    day = '{:02d}'.format(nowTimestamp.day)

    date_limit = str(year) + "-" + str(month) + "-" + str(day)
    hour_limit = str(nowTimestamp.round('h').hour) + ':' + "00"

    if request.POST.get('arima_pred'):
        data_arima = request.POST.get('date_arima')
        time_arima = request.POST.get('time_arima')
        predict_range_arima = request.POST.get('predict_range_arima')
        if predict_range_arima == '1 Hour':
            len_arima = 1+1
        elif predict_range_arima == '1 Day':
            len_arima = 24+1
        elif predict_range_arima == '1 Week':
            len_arima = 24 * 7+1
        else:
            len_arima = 1+1
        chart_arima = True
        #generate a pandas datetime for the query and start and end of the training set
        queryTS = bf.encode_query(data_arima,time_arima)
        # get data that model needs
        train_label, test_label, train_features, test_features, test_end, test_non_label = bf.get_exogenous_features(queryTS, len_arima, features)
        # predict
        predict_values = bf.AMI_ARIMA(train_label, test_label, train_features, test_features) # predict_values is dataframe with date as index

        train_arima = train_label.iloc[-24*2:-1]
        test_list_value = test_non_label.reset_index(drop=True).values.tolist()
        plot_data = [['NaN']*46, test_list_value]
        plot_arima = [item for sublist in plot_data for item in sublist]
        timesteps_arima = pd.date_range(start=train_arima.index[0], end=test_end+pd.to_timedelta('1h'), freq='H' )
        nanlist = ['NaN']*len(timesteps_arima[timesteps_arima <= train_arima.index[-1]])
        list_value = predict_values.reset_index(drop=True).values.tolist()

        predlist = [item for sublist in list_value for item in sublist]
        pred_arima = nanlist + predlist
    
        if chart_trf == False:
            #collect data to be sent to the frontend
            context = {
            'chart_arima': chart_arima,
            'chart_trf': chart_trf,
            'date_limit': date_limit,
            'hour_limit': hour_limit,

            'data_arima': data_arima,
            'time_arima': time_arima,
            'train_data_arima': train_arima.value.to_list(),
            "plot_arima":  plot_arima,
            'timesteps_arima': timesteps_arima.strftime('%Y-%m-%d %H:%M').to_list(),
            'pred_arima': pred_arima,
            'predict_range_arima' : predict_range_arima,

            }
        else:
            context = {
            'chart_arima': chart_arima,
            'chart_trf': chart_trf,
            'date_limit': date_limit,
            'hour_limit': hour_limit,

            'data_arima': data_arima,
            'time_arima': time_arima,
            'train_data_arima': train_arima.value.to_list(),
            "plot_arima": plot_arima,
            'timesteps_arima': timesteps_arima.strftime('%Y-%m-%d %H:%M').to_list(),
            'pred_arima': pred_arima,
            'predict_range_arima': predict_range_arima,

            'date_trf': data_trf,
            'time_trf': time_trf,
            'train_data_trf': train_trf.value.to_list(),
            "plot_trf": plot_trf,
            'timesteps_trf': timesteps_trf.strftime('%Y-%m-%d %H:%M').to_list(),
            'pred_trf': pred_trf,
            'predict_range_trf': predict_range_trf,
                
            }
        return render(request, 'ami/diplay_all.html', context)
    elif request.POST.get('trf_pred'):

        data_trf = request.POST.get('date_trf')
        time_trf = request.POST.get('time_trf')
        predict_range_trf = request.POST.get('predict_range_trf')
        if predict_range_trf == '1 Hour':
            len_trf = 1+1
        elif predict_range_trf == '1 Day':
            len_trf = 24+1
        elif predict_range_trf == '1 Week':
            len_trf = 24 * 7+1
        else:
            len_trf = 1+1
        train_label, predict_values, test_end, test_non_label = bf.transformer_predict(data_trf, time_trf, len_trf, model)
        chart_trf = True
        train_trf = train_label.iloc[-24*2:-1]
        test_list_value = test_non_label.reset_index(drop=True).values.tolist()
        plot_data = [['NaN'] * 46, test_list_value]
        plot_trf = [item for sublist in plot_data for item in sublist]
        timesteps_trf = pd.date_range(start=train_trf.index[0], end=test_end+pd.to_timedelta('1h'), freq='H' )

        nanlist = ['NaN']*len(timesteps_trf[timesteps_trf <= train_trf.index[-1]])
        list_value = predict_values.reset_index(drop=True).values.tolist()
        predlist = [item for sublist in list_value for item in sublist]
        pred_trf = nanlist + predlist
    
        #collect data to be sent to the frontend
        if chart_arima == False:
            context = {
                'chart_arima': chart_arima,
                'chart_trf': chart_trf,
                'date_limit': date_limit,
                'hour_limit':hour_limit,

                'date_trf': data_trf,
                'time_trf': time_trf,
                'train_data_trf': train_trf.value.to_list(),
                "plot_trf": plot_trf,
                'timesteps_trf': timesteps_trf.strftime('%Y-%m-%d %H:%M').to_list(),
                'pred_trf': pred_trf,
                'predict_range_trf': predict_range_trf,
                }
        else:
            context = {
                'chart_arima': chart_arima,
                'chart_trf': chart_trf,
                'date_limit': date_limit,
                'hour_limit': hour_limit,

                'data_arima': data_arima,
                'time_arima': time_arima,
                'train_data_arima': train_arima.value.to_list(),
                "plot_arima": plot_arima,
                'timesteps_arima': timesteps_arima.strftime('%Y-%m-%d %H:%M').to_list(),
                'pred_arima': pred_arima,
                'predict_range_arima': predict_range_arima,


                'date_trf': data_trf,
                'time_trf': time_trf,
                'train_data_trf': train_trf.value.to_list(),
                "plot_trf": plot_trf,
                'timesteps_trf': timesteps_trf.strftime('%Y-%m-%d %H:%M').to_list(),
                'pred_trf': pred_trf,
                'predict_range_trf': predict_range_trf,
                }
        return render(request, 'ami/diplay_all.html',context)



