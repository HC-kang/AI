# import warnings
# warnings.filterwarnings('ignore')

# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread('/Users/heechankang/projects/pythonworkspace/git_study/AI/data/dog.jpg', cv2.IMREAD_COLOR)

# img = cv2.resize(img, dsize = (224, 224))
# img = img.reshape((1, 224, 224, 3))

# from keras.applications.vgg16 import VGG16
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet_v2 import ResNet50V2

# model = VGG16()
# # model = InceptionV3()
# # model = ResNet50V2

# model.summary()

##############
import warnings
warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/Users/heechankang/projects/pythonworkspace/git_study/AI/data/dog.jpg', cv2.IMREAD_COLOR)

img = cv2.resize(img, dsize = (224, 224))
img = img.reshape((1, 224, 224, 3))

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet50V2

model = VGG16()
model = InceptionV3()
# model = ResNet50V2

model.summary()