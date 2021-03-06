import pathlib
import matplotlib.pyplot
import numpy
import pandas
import seaborn
import os

# the verbosity of tensorflow, this is on top as it needs to be defined before we import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
import tensorflow_docs
from src.epochdots import EpochDots
from src.plots import HistoryPlotter

dataset_path = tensorflow.keras.utils.get_file(
    "auto-mpg.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

EPOCHS = 1000
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
print('Static column mapping in data', column_names)

raw_dataset = pandas.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()
print(dataset.tail())

print(dataset.isna().sum())

dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
dataset = pandas.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print('Got test data and train data, here is a tail of the test data:')
print(test_dataset.tail())

plot = seaborn.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
matplotlib.pyplot.show()

# overall statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")  # this we want to guess ourselve
train_stats = train_stats.transpose()
print(train_stats)

# get training and verifying data
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

print('Next we are going to normalize the data so it is all in the same range of numbers,\n')
print(' this makes the data easier to train a network with, it also abstracts the units being used away')
print(' from the end model that we can save')


def normalize(x):
    """
    Do normalization on a dataset (table) using the statistics pandas series
    see return value of 'pandas.describe'

    :param x: pandas datastructure
    :return: normalized data
    """
    return (x - train_stats['mean']) / train_stats['std']


normalized_train_data = normalize(train_dataset)
normalized_test_data = normalize(test_dataset)

print(normalized_train_data)
assert not numpy.any(numpy.isnan(normalized_train_data))

print('This normalization we will also need to do to the live data in production before using this model')

print('Lets build the model')


def build_model():
    """
    :return tensorflow.keras.Sequential:
    """
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(1)
    ])

    optimizer = tensorflow.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


model = build_model()
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

print(model.summary())

example_batch = normalized_train_data[:10]
example_result = model.predict(example_batch)
print('Predictions from the untrained model:', example_result)

history = model.fit(
    normalized_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop, EpochDots()]
)

history_dataframe = pandas.DataFrame(history.history)

print()
print(history_dataframe.tail())

# clear the current plots first
matplotlib.pyplot.clf()

plotter = HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric='mae')
matplotlib.pyplot.ylim([0,20])
matplotlib.pyplot.ylabel('MAE [MPG^2]')
matplotlib.pyplot.show()

plotter.plot({'Basic': history}, metric = "mse")
matplotlib.pyplot.ylim([0, 20])
matplotlib.pyplot.ylabel('MSE [MPG^2]')
matplotlib.pyplot.show()

print('Lets see how good the model works when applying the test data for validation')
loss, mae, mse = model.evaluate(normalized_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normalized_test_data).flatten()

print("Lets plot the predictions vs the labels")
matplotlib.pyplot.clf()
a = matplotlib.pyplot.axes(aspect='equal')
matplotlib.pyplot.scatter(test_labels, test_predictions)
matplotlib.pyplot.xlabel('True Values [MPG]')
matplotlib.pyplot.ylabel('Predictions [MPG]')
lims = [0, 50]
matplotlib.pyplot.xlim(lims)
matplotlib.pyplot.ylim(lims)
matplotlib.pyplot.plot(lims, lims)
matplotlib.pyplot.show()

error = test_predictions - test_labels
matplotlib.pyplot.hist(error, bins = 25)
matplotlib.pyplot.xlabel("Prediction Error [MPG]")
matplotlib.pyplot.ylabel("Count")
matplotlib.pyplot.show()