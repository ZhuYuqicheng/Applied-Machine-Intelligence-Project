import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from tqdm import tqdm


class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = StandardScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # return number of sensors
        return len(self.gdf.roupby(by=["reindexed_id"]))

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        
        # Sensors are indexed from 1
        idx = idx+1

        # np.random.seed(0)

        start = np.random.randint(0, len(self.df[self.df["reindexed_id"]==idx]) - self.T - self.S) 
        sensor_number = str(self.df[self.df["reindexed_id"]==idx][["sensor_id"]][start:start+1].values.item())
        index_in = torch.tensor([i for i in range(start, start+self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        _input = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)
        target = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform

        scaler.fit(_input[:,0].unsqueeze(-1))
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return index_in, index_tar, _input, target, sensor_number


class SpotPriceDataset(Dataset):

    def __init__(self, data_list_path, training_length, forecast_window, test=False):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """

        # load data
        super(SpotPriceDataset, self).__init__()
        self.data_list = np.loadtxt(data_list_path, skiprows=1, dtype=str)
        self.T = training_length
        self.S = forecast_window
        # self.data_time = []
        self.data_value = []
        self.transform = MinMaxScaler()

        for row in tqdm(self.data_list, "Loading dataset into memory"):

            data = row.split(",")
            # data_date = data[0]
            # data_hour = int(float(data[1]))
            # self.data_time.append(data_date + '-' + str(data_hour))

            # data_val = (np.array(data[2:25]))
            data_val = [float(x) for x in data if x != '']
            self.data_value.append(data_val)
        self.data_value = np.asarray(self.data_value)
        # self.transform.fit(self.data_value[:, 0].reshape(-1, 1))
        # if not test:
        #     dump(self.transform, 'scalar_item.joblib')
        # else:
        #     dump(self.transform, 'scalar_item_test.joblib')

    def __len__(self):
        # return number of data
        return len(self.data_list) - self.T - self.S

    # index between 0 and __len__.
    def __getitem__(self, idx):
        """
        param idx: int
        return:
        """

        # idx means where to start the sequence

        # np.random.seed(0)
        start = idx
        index_input = torch.tensor(list(range(start, start + self.T)))
        index_to_pre = torch.tensor(list(range(start + self.T, start + self.T + self.S)))

        _input = torch.tensor(self.data_value[start: start + self.T])
        target = torch.tensor(self.data_value[start + self.T: start + self.T + self.S])

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].

        self.transform.fit(_input[:, 0].unsqueeze(-1))
        _input[:, 0] = torch.tensor(self.transform.transform(_input[:, 0].unsqueeze(-1)).squeeze(-1))
        target[:, 0] = torch.tensor(self.transform.transform(target[:, 0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(self.transform, 'scalar_item.joblib')

        return index_input, index_to_pre, _input, target