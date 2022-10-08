import pandas as pd
import sys

import torch
from sklearn.preprocessing import MinMaxScaler

import statsmodels.formula.api as smf

import torch 
import torch.nn as nn

class helo():
    def test(self):
        print('hello')

def load_data_2010_2020():
    bulan2id = {
        "Januari": 1,
        "Februari": 2,
        "Maret": 3,
        "April": 4,
        "Mei": 5,
        "Juni": 6,
        "Juli": 7,
        "Agustus": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Desember": 12
    }
     # Data untuk 2010 - 2020
    data = pd.read_csv('./datasets/bps_harga_beras.csv', sep =';')
    # Mengambil nama kolom di dataset
    print(data.keys())
    # Mengambil nilai / value di kolom tertentu
    beras_data = data[['nama_tahun', 'nama_turunan_tahun', 'data_content']]

    beras_data = beras_data.rename(columns={"nama_tahun": "tahun", "nama_turunan_tahun": "bulan", "data_content": "harga_beras"})
    beras_data["harga_beras"] = beras_data["harga_beras"].round()
    beras_data.bulan =  beras_data.bulan.map(bulan2id)
    
    # beras_data.append({"tahun": 2020, "bulan": 8, "harga_beras": beras_data[:1, ["harga_beras"]]})
    print(beras_data)

def load_data(tahun):
    bulan2id = {
        "Januari": 1,
        "Februari": 2,
        "Maret": 3,
        "April": 4,
        "Mei": 5,
        "Juni": 6,
        "Juli": 7,
        "Agustus": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Desember": 12
    }

    # Untuk load file excel untuk data 2021
    beras_data = pd.read_excel('./datasets/data_beras_'+ str(tahun) +'.xlsx')
    # untuk transpose / putar dari horizontal ke vertikal bentuk table nya
    beras_data_raw = beras_data[:5].T
    beras_data_raw = beras_data_raw[1:13]
    beras_data_raw = beras_data_raw.loc[:, 1:4]
    
    beras_data_raw = beras_data_raw.rename(columns={1: "bulan", 2: "premium", 3: "medium", 4: "luar_kualitas"})
    beras_data_raw["premium"] = pd.to_numeric(beras_data_raw["premium"])
    beras_data_raw["medium"] = pd.to_numeric(beras_data_raw["medium"])
    beras_data_raw["luar_kualitas"] = pd.to_numeric(beras_data_raw["luar_kualitas"])
    beras_data_raw["tahun"] = tahun
    # Mean dengan axis 1 dia rata rata horizontal  (kiri ke kanan)
    # Mean dengan axis 0 dia rata rata vertical (atas ke bawah)
    beras_data_raw["harga_beras"] = beras_data_raw[["premium", "medium", "luar_kualitas"]].mean(axis=1).round()
    beras_data_raw = beras_data_raw[["tahun", "bulan", "harga_beras"]].reset_index(drop=True)
    beras_data_raw.bulan =  beras_data_raw.bulan.map(bulan2id)
    
    return beras_data_raw
    

def load_data_bbm():
    data_bbm = pd.read_csv("./datasets/harga_bbm.csv")
    data_bbm = data_bbm.rename(columns={1: "tahun", 2:"bulan", 3:"harga_bbm"})
    return data_bbm

def repair_digit_harga_bbm(x):
    x = x.replace(".", "")
    x = x.replace(",", ".")
    return x

def preprocess():
    # lOAD semua data beras
    data_19 = load_data(2019)
    data_20 = load_data(2020)
    data_21 = load_data(2021)

    data_beras = pd.concat([data_19, data_20, data_21]).reset_index(drop=True)

    # Load data harga_bbm
    data_bbm = load_data_bbm()
    
    datasets = pd.concat([data_beras, data_bbm["harga_bbm"]], axis=1)

    datasets["harga_bbm"] = datasets["harga_bbm"].apply(lambda x: f"{repair_digit_harga_bbm(x)}")
    datasets["harga_bbm"] = pd.to_numeric(datasets["harga_bbm"])

    data_bbm = datasets[["tahun", "bulan", "harga_bbm"]]
    data_beras = datasets[["tahun", "bulan", "harga_beras"]]

    data_beras["prev_harga_beras"] = data_beras["harga_beras"].shift(1)
    data_bbm["prev_harga_bbm"] = data_bbm["harga_bbm"].shift(1)

    data_beras["diff_beras"] = (data_beras["harga_beras"] - data_beras["prev_harga_beras"])
    data_bbm["diff_bbm"] = (data_bbm["harga_bbm"] - data_bbm["prev_harga_bbm"])

    for bln in range(1, 13):
        field_name = "lag_" + str(bln)
        data_beras["beras_" + field_name] = data_beras["diff_beras"].shift(bln)
        data_bbm["bbm_" + field_name] = data_bbm["diff_bbm"].shift(bln)

    train_beras_set = data_beras[0: -6].values
    test_beras_set = data_beras[-6:].values

    train_beras_set = data_beras[-6:].values
    scaler = MinMaxScaler(feature_range = (-1, 1))
    scaler = scaler.fit(train_beras_set)

    train_beras_set = train_beras_set.reshape(train_beras_set.shape[0], train_beras_set.shape[1])

    print(train_beras_set.shape)

    train_set_scaled = scaler.transform(train_beras_set)

    test_beras_set = test_beras_set.reshape(test_beras_set.shape[0], test_beras_set.shape[1])
    test_set_scaled = scaler.transform(test_beras_set)

    x_train, y_train = train_set_scaled[:, 1:],  train_set_scaled[:, 0:1]
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

    x_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()

    return x_train, y_train, x_test, y_test, scaler, train_set_scaled, test_set_scaled 

    # print(train_set_scaled)

    # print(scaler)

if __name__ == '__main__':
    # Memanggil method di class
    h = helo()
    h.test()

    # Memanggil method
    x_train, y_train, x_test, y_test, scaler, train_set_scaled, test_set_scaled = preprocess()

