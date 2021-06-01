import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/AI/')

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
model = load_model('/Users/heechankang/projects/pythonworkspace/git_study/AI/model/model_cifar10.h5')

model.summary()

import cv2
import matplotlib.pyplot as plt


img = cv2.imread('/Users/heechankang/projects/pythonworkspace/git_study/AI/data/automobile.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('/Users/heechankang/projects/pythonworkspace/git_study/AI/data/dog.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('/Users/heechankang/projects/pythonworkspace/git_study/AI/data/cat.1.jpg', cv2.IMREAD_COLOR)


plt.imshow(img, interpolation="bicubic")
plt.show()

img.shape

new_img = cv2.resize(img, dsize = (32,32))

plt.imshow(new_img, interpolation = 'bicubic')
plt.show()

new_img.shape

new_img = new_img.reshape(1, 32, 32, 3)

# 스케일링 추가 필요
result = model.predict_classes(new_img)

label_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

result = label_name.get(result[0])

result