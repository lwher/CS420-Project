# Train Fer2013 Classifiers

We play with sklearn and Pytorch on the Fer2013 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- sklearn 0.20.3

## Accuracy
| Model             | Test Acc.   |
| ----------------- | ----------- |
| KNN               | 41.42%      |
| Decision Tree     | 29.65%      |
| RandomForest      | 36.67%      |
| SVM(rbf)          | 43.30%      |
| MLP               | 43.63%      |
| LeNet             | 55.44%      |
| ResNet18          | 63.61%      |
| ResNet50          | 63.70%      |
| ResNeXt29 32x4d   | 62.97%      |
| ResNeXt29 2x64d   | 64.50%      |
| VGG16             | 64.59%      |
| ShuffleNetV2 1x   | 65.00%      |
| MobileNetV2       | 65.39%      |
| GoogleNet         | 66.93%      |
| DenseNet121       | 68.26%      |

## Learning rate adjustment
We manually change the `lr` during training:
- `0.1` for epoch `[0,40)`
- `0.01` for epoch `[40,60)`
- `0.001` for epoch `[60,80)`

First, use split_data.py to split the dataset `fer2013.csv` into `train.csv`, `val.csv` and `test.csv`

Then, put these three data files into data folder

Start training traditional methods with `python xxx.py` like `python SVM.py`

Start training deep learning methods with `python main.py`
Resume the deep training with `python main.py --resume --lr=0.01`
