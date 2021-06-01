import warnings
warnings.filterwarnings('ignore')

from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images.shape

train_labels.shape

test_images.shape

test_labels.shape

train_images.size

import matplotlib.pyplot as plt

for i in range(0, 6):
    img = train_images[i]
    plt.imshow(img, interpolation='bicubic')
    plt.show()