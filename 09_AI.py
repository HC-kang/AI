import warnings
warnings.filterwarnings('ignore')

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 네트워크 모델 설정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (28*28,)))
model.add(Dense(10, activation = 'softmax'))

from keras.layers.advanced_activations import LeakyReLU
activation = LeakyReLU(.001)
model = Sequential()
model.add(Dense(512, activation = activation, input_shape = (28*28,)))
model.add(Dense(10, activation = 'softmax'))

# RMSprop
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop(lr = 0.001)
    # test_acc with relu : 0.978900015354156
    # test_acc with LeakyReLU : 0.978600025177002


# Adam
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.01)
    # test_acc with relu : 0.9628000259399414
    # test_acc with LeakyReLU : 0.9670000076293945

        # 둘 다 유의미한 차이는 없는것같은데..?

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = 'accuracy')

model.summary()

# 모델 훈련
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc :' , test_acc)