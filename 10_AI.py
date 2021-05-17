import warnings
warnings.filterwarnings('ignore')

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## 네트워크 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Activation - 하이퍼볼릭 탄젠트
model = Sequential()
model.add(Dense(128, activation = 'tanh', input_shape = (28*28,)))
model.add(Dense(10, activation = 'softmax'))
    # test_acc : 0.9739999771118164

# Activation - 리키렐루
from keras.layers.advanced_activations import LeakyReLU
model = Sequential()
activation = LeakyReLU(.001)
    # test_acc : 0.9801999926567078

# Activation - 엘루
from keras.layers.advanced_activations import ELU
model = Sequential()
activation = ELU(.001)
    # test_acc : 0.982200026512146


model.add(Dense(512, activation = activation, input_shape = (28*28,)))
model.add(Dense(10, activation = 'softmax'))

# Optimizer - 아담
from tensorflow.keras.optimizers import Adam
optimizer = Adam(.001)

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = 'accuracy')

model.summary()


# 모델 훈련
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc :', test_acc)
