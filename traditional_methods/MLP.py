import numpy as np
import os
import sys

from data_loader import LoadDataset
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

'''
Dim: 256
Model             Val            Test
MLP(256, 256)    43.72%         43.63%
'''

def FeatureReduction(train_data, val_data, test_data, dimension):
    pca_class = PCA(n_components=dimension)
    pca_model = pca_class.fit(train_data)
    train_data = pca_model.transform(train_data)
    val_data = pca_model.transform(val_data)
    test_data = pca_model.transform(test_data)
    return train_data, val_data, test_data

def TestModel(model, data, label, dataset_name):
    print('Testing on', dataset_name)
    print(model.score(data, label))

def main():
    # prepare data
    print('Start loading data...')
    train_data, train_label, val_data, val_label, test_data, test_label = LoadDataset()
    train_data = (train_data * 1.0 / 128.0) - 1
    val_data = (val_data * 1.0 / 128.0) - 1
    test_data = (test_data * 1.0 / 128.0) - 1
    train_data = train_data.reshape(-1, 48 * 48)
    val_data = val_data.reshape(-1, 48 * 48)
    test_data = test_data.reshape(-1, 48 * 48)
    print('Finish loading data.')
    print('train data shape', train_data.shape)
    print('val data shape', val_data.shape)
    print('test data shape', test_data.shape)

    train_data, val_data, test_data = FeatureReduction(train_data, val_data, test_data, 256)

    print('After PCA')
    print('train data shape', train_data.shape)
    print('val data shape', val_data.shape)
    print('test data shape', test_data.shape)

    # train_model
    print('Start training...')
    model = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(256,256), random_state=1,max_iter=100,verbose=10,learning_rate_init=.05)
    model.fit(train_data, train_label)
    print('Training finished.')
    TestModel(model, val_data, val_label, 'val_set')
    TestModel(model, test_data, test_label, 'test_set')

if __name__ == '__main__':
    main()