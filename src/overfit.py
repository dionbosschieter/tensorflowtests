import os

# the verbosity of tensorflow, this is on top as it needs to be defined before we import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot
import numpy
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

gzipped_data = tf.keras.utils.get_file('HIGGS.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')

FEATURES = 28

data = tf.data.experimental.CsvDataset(gzipped_data, [float(),]*(FEATURES+1), compression_type='GZIP')

def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label

packed_data = data.batch(10000).map(pack_row).unbatch()

for features, label in packed_data.batch(1000).take(1):
    print(features[0])
    pyplot.hist(features.numpy().flatten(), bins = 101)