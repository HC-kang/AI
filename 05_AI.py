import warnings
warnings.filterwarnings('ignore')

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
test_images.shape
train_labels.shape
test_images.shape
        # 데이터 불러오기

train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

train_images.shape
test_images.shape
        # 데이터 전처리

digit = train_images[0]
digit
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images[0]
        # 정규화

train_labels[0]
train_labels.shape

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_labels[0]
train_labels.shape

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation = 'relu', input_dim = 784))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'RMSProp',
              metrics = 'accuracy')

model.summary()

model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)


###########
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/AI')
if not os.path.exists('./model'):
    os.mkdir('./model')
model.save('model/model.h5')

#################

import warnings
warnings.filterwarnings('ignore')

from keras.models import load_model

model = load_model('model/model.h5')
model.summary()

from keras.datasets import mnist
import matplolib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

pick_img = test_images[25]
plt.imshow(pick_img, cmap=plt.cm.binary)
plt.show()