import os

# the verbosity of tensorflow, this is on top as it needs to be defined before we import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy
import tensorflow
from tensorflow import keras
import tensorflow_datasets
import matplotlib.pyplot

print(tensorflow.__version__)

BUFFER_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 10

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

for train_example, train_label in train_data.take(1):
  print('Encoded text =', train_example.numpy())
  print('Decoded text =', encoder.decode(train_example))
  print('Label =', train_label.numpy())

print('Lets get the current shapes that is perhaps a bit easier')

# the none here allows for 'dynamic' padding, instead of guessing or getting max the size
padded_shapes = ((None,), ())

train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes))
test_batches = (test_data.padded_batch(BATCH_SIZE, padded_shapes))

for example_batch, label_batch in train_batches.take(2):
    print('Batch shape = ', example_batch.shape)
    print('label shape = ', label_batch.shape)

print('Lets build a continuous bag of words style model')
model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Embedding(encoder.vocab_size, 16),
    tensorflow.keras.layers.GlobalAveragePooling1D(),
    tensorflow.keras.layers.Dense(1)
])

model.summary()

print('We are going to compile the model')
model.compile(optimizer='adam',
              loss=tensorflow.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('Lets train the model')
history = model.fit(train_batches, epochs=EPOCHS, validation_data=test_batches, validation_steps=30)

loss, accuracy = model.evaluate(test_batches, verbose=2)
print('Loss =', loss)
print('Accuracy =', accuracy)

history_dict = history.history
print('Got a history dict with the following keys =', history_dict.keys())

print('Lets plot these dict values')

x_axes = range(1, EPOCHS + 1)
for key in history_dict.keys():
    y_axes = numpy.array(history_dict[key]) * 100
    matplotlib.pyplot.plot(x_axes, y_axes, 'b', color=numpy.random.rand(3, ), label=key)

matplotlib.pyplot.title('Training history')
matplotlib.pyplot.xlabel('Training loops')
matplotlib.pyplot.ylabel('Percentage')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()