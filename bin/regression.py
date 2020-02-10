import pathlib
import matplotlib
import numpy
import pandas
import seaborn
import os

# the verbosity of tensorflow, this is on top as it needs to be defined before we import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
import tensorflow_docs

dataset_path = tensorflow.keras.utils.get_file(
    "auto-mpg.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
print('Static column mapping in data', column_names)

raw_dataset = pandas.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

print(dataset.isna().sum())

dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
dataset = pandas.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print('Got test data and train data, here is a tail of the test data:')
print(test_dataset.tail())

plot = seaborn.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plot.savefig("/tmp/fig.png")

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

print(train_stats)
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

