import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

print('Loaded training and test data')

train_examples_batch, train_labels_batch = next(iter(train_data.batch(1)))
# print(train_examples_batch)
print(train_labels_batch)

embedding_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
print('Lets use a pre trained model from', embedding_url)

hub_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)

print('Printing the retrieved hub layer')
print(hub_layer(train_examples_batch[:3]))

print('Adding pre-trained model to our model')
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print('Model summary:')
print(model.summary())

print('Lets compile the model')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Lets train the model')
model.fit(
    train_data.shuffle(10000).batch(512),
    epochs=10,
    validation_data=validation_data.batch(512)
)

test_loss, test_accuracy = model.evaluate(test_data.batch(512), verbose=2)
print('The accuracy reached =', test_accuracy)
print('The loss     reached =', test_loss)

print('Lets verify the model, getting one item from the test data')

list_of_verify_data = train_data.batch(1)
print(list_of_verify_data)
verify_text_batch, verify_labels_batch = next(iter(list_of_verify_data))
print('Working with the following data')
print(verify_text_batch)
print('Expected label =', verify_labels_batch)
prediction = model.predict(list_of_verify_data)[0]
print('Prediction results for the data =', prediction)
