import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def data_provider(configs):
    data_set = ETFDataset(configs)
    print('Length:', len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.num_workers,
        drop_last=True)
    return data_set, data_loader


class ETFDataset(Dataset):
    def __init__(self, configs):
        self.train_len = configs.train_len # train window
        self.test_len = configs.test_len # test window
        self.cov_window = configs.cov_window # covariance estimation window
        self.horizon = configs.horizon # prediction horizon
        self.stride = configs.stride if configs.stride else self.test_len
        self.input_dim = configs.input_dim
        self.min_periods = max(self.input_dim, self.cov_window)  # minimum estimation period

        self.data_path = configs.data_path
        self.__read_data__()
        
    def __read_data__(self):
        df = pd.read_csv(self.data_path, header=None, usecols=[0, 1, 2])
        df.columns = ['RIC', 'Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['Return'] = df.groupby('RIC')['Price'].pct_change(fill_method=None)
        df.dropna(subset=['Return'], inplace=True)
        self.data = df.pivot(index='Date', columns='RIC', values='Return').fillna(0).values
    
    def __getitem__(self, index):
        """
        x: [T, N], historical returns
        y: [H, N], next H period returns
        """
        T = len(self.data)

        train_start = index * self.stride + self.min_periods
        train_end   = train_start + self.train_len
        test_start  = train_end

        def _eff_len(start, want_len):
            max_len = T - start - self.horizon + 1
            return max(0, min(want_len, max_len))

        train_len_eff = _eff_len(train_start, self.train_len)
        test_len_eff  = _eff_len(test_start,  self.test_len)

        if test_len_eff <= 0:
            raise IndexError("Index out of bounds for the dataset length (no room for horizon).")

        x0_train = torch.stack([
            torch.tensor(self.data[train_start + k - self.cov_window:train_start + k, :], dtype=torch.float32)
            for k in range(train_len_eff)
        ], dim=0)
        x_train = torch.stack([
            torch.tensor(self.data[train_start + k - self.input_dim:train_start + k, :], dtype=torch.float32)
            for k in range(train_len_eff)
        ], dim=0)
        y_train = torch.stack([
            torch.tensor(self.data[train_start + k:train_start + k + self.horizon], dtype=torch.float32)
            for k in range(train_len_eff)
        ], dim=0)

        x0_test = torch.stack([
            torch.tensor(self.data[test_start + k - self.cov_window:test_start + k, :], dtype=torch.float32)
            for k in range(test_len_eff)
        ], dim=0)
        x_test = torch.stack([
            torch.tensor(self.data[test_start + k - self.input_dim:test_start + k, :], dtype=torch.float32)
            for k in range(test_len_eff)
        ], dim=0)
        y_test = torch.stack([
            torch.tensor(self.data[test_start + k:test_start + k + self.horizon], dtype=torch.float32)
            for k in range(test_len_eff)
        ], dim=0)

        test_end_eff = test_start + test_len_eff

        return x0_train, x_train, y_train, x0_test, x_test, y_test, test_start, test_end_eff

    def __len__(self):
        return (len(self.data) - self.train_len - self.test_len - self.min_periods - self.horizon) // self.stride + 1
