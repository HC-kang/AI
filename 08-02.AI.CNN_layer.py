import warnings
warnings.filterwarnings('ignore')

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), input_shape = ( 28, 28, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.001)

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = 'accuracy')
              
model.summary()

model.fit(train_images, train_labels, epochs = 5, batch_size = 200)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

