#####
# batch normalization
###

import warnings
warnings.filterwarnings('ignore')

from keras.datasets import mnist
from keras.utils import to_categorical

import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras import regularizers
reg = regularizers.l2(0.01)

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

model = Sequential()

model.add(Dense(128, input_shape = (784, )))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64, input_shape = (512, )))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, activation = 'softmax'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.001)

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = 'accuracy')
              
model.summary()

model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print( 'test acc: ', test_acc)

