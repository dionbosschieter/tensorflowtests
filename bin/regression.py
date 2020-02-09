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