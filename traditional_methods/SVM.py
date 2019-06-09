import numpy as np
import os
import sys

from data_loader import LoadDataset
from sklearn import svm
from sklearn.decomposition import PCA

'''
Dim: 256
60% Training Data

Model             Val            Test
SVM(linear)      34.98%         36.86%
SVM(rbf)         42.46%         43.30%
SVM(poly)        36.67%         37.28%
'''

def TrainSVM(train_data, train_label, train_kernel='linear'):
    classifier = svm.SVC(C=1.0, kernel=train_kernel, gamma='scale', decision_function_shape='ovr')
    classifier.fit(train_data, train_label)
    return classifier

def TestModel(classifier, test_data, test_label, dataset_name):
	print('Testing on', dataset_name)
	print('Accuracy', classifier.score(test_data, test_label))

def FeatureReduction(train_data, val_data, test_data, dimension):
    pca_class = PCA(n_components=dimension)
    pca_model = pca_class.fit(train_data)
    train_data = pca_model.transform(train_data)
    val_data = pca_model.transform(val_data)
    test_data = pca_model.transform(test_data)
    return train_data, val_data, test_data

def main():
	# prepare data
	print('Start loading data...')
	train_data, train_label, val_data, val_label, test_data, test_label = LoadDataset()
	train_data = (train_data * 1.0 / 128.0) - 1
	val_data = (val_data * 1.0 / 128.0) - 1
	test_data = (test_data * 1.0 / 128.0) - 1
	train_data = train_data.reshape(-1, 48 * 48)
	train_data = train_data[:18000, :]
	train_label = train_label[:18000]
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
	model = TrainSVM(train_data, train_label, 'poly')
	print('Training finished.')
	TestModel(model, val_data, val_label, 'val_set')
	TestModel(model, test_data, test_label, 'test_set')

if __name__ == '__main__':
    main()