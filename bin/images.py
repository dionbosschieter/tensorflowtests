from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# the verbosity of tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_image(predictions_array, img):
    """
    plot a image with a label underneath showing the possible matches
    :param index: index
    :param predictions_array: array with the predictions
    :param img:
    :return:
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    label_as_text = static_clothing_labels[predicted_label]

    # add a label to the end
    plt.xlabel(
        "{} {:2.0f}%".format(label_as_text, 100 * np.max(predictions_array)),
        color='black'
    )


def plot_value_array(predictions_array):
    """
    plot a bar graph with the predictions and their probability
    :param predictions_array:
    :return:
    """
    predictions_array = predictions_array
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

static_clothing_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('We have', train_images.shape[0], 'images in total')
print('We have', len(train_labels), 'labels in total')
print(static_clothing_labels)

print('We have', test_images.shape[0], 'test images in total')
print('We have', len(test_labels), 'test labels in total')

print('We will first train the model')
print('Then we will evaluate against the test data')
print('To see how effective the trained model actually is')

print('Showing the first image')
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

print('We are transforming the test and train images to a black'
      ' and white model by removing the color by doing a divide with 255')
train_images = train_images / 255.0
test_images = test_images / 255.0

print('Lets manually verify this by showing the first 25 images')
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(static_clothing_labels[train_labels[i]])
plt.show()

print('Defining the model, first we flatten it to one line,',
      'then we dense with 128 neurons and at last',
      'with 10 node layer that results in an array of',
      '10 probabiliy scores so we know what the curent image belongs to')

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

print('Compile the model, this is needed before using it')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training the model, this could take some time...')
model.fit(train_images, train_labels, epochs=5)

print('Next we are going to test the just trained model, using our test data')
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print('The accuracy reached =', test_accuracy)
print('The loss     reached =', test_loss)

img = test_images[1]
print(img.shape)
img = (np.expand_dims(img, 0))  # we will need to make a array with only one image in it
prediction = model.predict(img)[0]
print('prediction results for the first image = ', prediction)
predicted_label = np.argmax(prediction)
if test_labels[1] == predicted_label:
    print('The prediction for the first image is correct!')

plot_value_array(prediction)
_ = plt.xticks(range(10), static_clothing_labels, rotation=45)

print('Now lets use the model to predict all of the test_images')
predictions = model.predict(test_images)

# printing the first image + bar graph image
# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(predictions[i], test_images[i])
# plt.subplot(1, 2, 2)
# plot_value_array(predictions[i])
# plt.show()

num_images = 25
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.floor(np.sqrt(num_images)))
num_images = num_rows * num_cols
print('We are showing the first', num_images, 'for you to verify')

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(predictions[i], test_images[i])
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(predictions[i])
plt.tight_layout()  # padding between plots
plt.show()
