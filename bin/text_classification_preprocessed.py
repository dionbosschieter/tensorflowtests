import os

# the verbosity of tensorflow, this is on top as it needs to be defined before we import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
from tensorflow import keras
import tensorflow_datasets

print(tensorflow.__version__)

(train_data, test_data), info = tensorflow_datasets.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    split=(tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    with_info=True
)

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_text = 'Hello TensorFlow.'
encoded_text = encoder.encode(sample_text)
print('Encoded text is {}'.format(encoded_text))

original_text = encoder.decode(encoded_text)
print('The original text is {}'.format(original_text))

for vec in encoded_text:
    print('{} = {}'.format(vec, encoder.decode([vec])))