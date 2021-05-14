import keras
from keras.datasets import mnist

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

type(train_images)

digit = train_images[4]
digit

plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

print(train_images.shape)

print(train_labels.shape)

train_labels

print(test_images.shape)