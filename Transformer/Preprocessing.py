import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic


# encoding the timestamp data cyclically. See Medium Article.
def process_data(source):
    df = pd.read_csv(source)

    timestamps = df['Date'].values
    timestamps_year = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').year) for t in timestamps])
    timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').day) for t in timestamps])
    timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').month) for t in timestamps])
    timestamps_hour = df['Time of day'].values

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    df['sin_hour'] = np.sin(2 * np.pi * timestamps_hour / hours_in_day)
    df['cos_hour'] = np.cos(2 * np.pi * timestamps_hour / hours_in_day)
    df['sin_day'] = np.sin(2 * np.pi * timestamps_day / days_in_month)
    df['cos_day'] = np.cos(2 * np.pi * timestamps_day / days_in_month)
    df['sin_month'] = np.sin(2 * np.pi * timestamps_month / month_in_year)
    df['cos_month'] = np.cos(2 * np.pi * timestamps_month / month_in_year)
    df['year'] = timestamps_year

    df.drop(['Date', 'Time of day'], axis=1, inplace=True)
    return df


# df = pd.read_csv('Data/data_pe.csv')
# df = df.iloc[35056+3623:-744]
# df = df[['Spot price', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']]
# df.to_csv(r'Data/data_train_1.csv', index=False)
