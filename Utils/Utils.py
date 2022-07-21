import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Get close data from csv file
def get_data(folder, src_name):
    path = os.path.join(folder, f"{src_name}.csv")
    df = pd.read_csv(path)
    data = df.close.to_numpy()
    return data

def load_time_data(folder, src_name, time_config):
    path = os.path.join(folder, f"{src_name}.csv")
    df = pd.read_csv(path)
    start_day = time_config["start-day"]
    finish_day = time_config["finish-day"]
    time_cut_df = df[(df['datetime'] > start_day) & (df['datetime'] < finish_day)]
    data = time_cut_df.close.to_numpy()
    return data


def time_series_processing(data, mode, setting):
    if mode == "one-day":
        n_sample = data.shape[0]
        seq_len = setting["seq_len"]
        x = [data[i : i+seq_len] for i in range(0, n_sample - seq_len)]
        y = [data[i] for i in range(seq_len, n_sample)]
        return {
            "X": np.array(x, dtype=np.float32),
            "Y": np.array(y, dtype=np.float32).reshape(-1, 1)
        }
            
    if mode == "test":
        seq_len = 22
        horizon = 3
        n_sample = data.shape[0]
        x = [data[i : i+seq_len] for i in range(0, n_sample-seq_len)]
        y = [data[i : i+horizon] for i in range(seq_len, n_sample-horizon)]
        n_sample = min(len(x), len(y))
        x = x[:n_sample]
        y = y[:n_sample]
        return {
            "X": np.array(x, dtype=np.float32),
            "Y": np.array(y, dtype=np.float32)
        }   
        
    if mode == "future-day":     
        n_sample = data.shape[0]
        seq_len = setting["seq_len"]
        future = setting["future"]
        x = [data[i : i+seq_len] for i in range(0, n_sample - seq_len)]
        # next day correspond with future = 1
        y = [data[i] for i in range(seq_len + future - 1, n_sample)]
        n_sample = min(len(x), len(y))
        x = x[:n_sample]
        y = y[:n_sample]
        return {
            "X": np.array(x, dtype=np.float32),
            "Y": np.array(y, dtype=np.float32).reshape(-1, 1)
        }    


def preprocessing(data, name, test_ratio, mode, setting):
    train = time_series_processing(data=data, mode=mode, setting=setting)
    # Scale data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(train["X"])
    y_scaler.fit(train["Y"])
    train["X"] = x_scaler.transform(train["X"])
    train["Y"] = y_scaler.transform(train["Y"])
    # test["X"] = x_scaler.transform(test["X"])
    # test["Y"] = y_scaler.transform(test["Y"])

    return {
        "name": name,
        "train": train,
        # "test": test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler
    }