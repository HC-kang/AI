#####
# regularizer
###

import warnings
warnings.filterwarnings

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical


# 데이터 로드
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 네트워크 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Regularizer - L2
from tensorflow.keras import regularizers
reg = regularizers.l2(0.01)

model = Sequential()
model.add(Dense(126, activation = 'relu', kernel_regularizer = reg, input_shape = (28*28,)))
model.add(Dense(10, activation = 'softmax'))

# Adam Optimizer
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.001)
model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = 'accuracy')

model.summary()

# 모델 훈련
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc : ', test_acc)
    # test_acc :  0.9553999900817871
