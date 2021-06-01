import warnings
warnings.filterwarnings('ignore')

from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from keras.layers.normalization import BatchNormalization

input = Input(shape = (32, 32,3))
model = ResNet50(input_tensor = input, include_top = False, weights=None, pooling = 'max')

x1 = model.output
x2 = Dense(1024)(x1)
x3 = Activation('relu')(x2)
x4 = Dense(512)(x3)
x5 = Activation('relu')(x4)
x = Activation('softmax')(x5)

model = Model(model.input,x)

model.summary()

