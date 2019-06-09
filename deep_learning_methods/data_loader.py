# -*- coding: utf-8 -*-
import csv
import os
import numpy as np

def LoadDataSubset(data_csv):
    csv_file = data_csv
    num = 1
    data_list = []
    label_list = []
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            pixels = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            data_list.append(pixels)
            label_list.append(label)
    data_list = np.array(data_list, dtype=np.float32)
    data_list = 1.0 * data_list / 128.0 - 1.0 
    label_list = np.array(label_list, dtype=np.int64)
    return data_list, label_list

def LoadDataset():
    # data path
    datasets_path = r'.\data'
    train_csv = os.path.join(datasets_path, 'train.csv')
    val_csv = os.path.join(datasets_path, 'val.csv')
    test_csv = os.path.join(datasets_path, 'test.csv')
    # load data
    train_data, train_label = LoadDataSubset(train_csv)
    val_data, val_label = LoadDataSubset(val_csv)
    test_data, test_label = LoadDataSubset(test_csv)
    return train_data, train_label, val_data, val_label, test_data, test_label
