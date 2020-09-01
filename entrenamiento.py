import sys
import os

from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD

data = []
labels = []
batchSize = 32

# def loadDataSet():




imageP = paths.list_images("./data/entrenamiento/Knives")
for i in imageP:

    image = cv2.imread(i)
    image = cv2.resize(image, (100, 100))
    image = img_to_array(image)
    data.append(image)

    labels.append(1)


imageP = paths.list_images("./data/entrenamiento/No_Knives")
for i in imageP:

    image = cv2.imread(i)
    image = cv2.resize(image, (100, 100))
    image = img_to_array(image)
    data.append(image)

    labels.append(0)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=25)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


# print('as')
# print(trainX)
# print(testX)
# print(trainY)
# print(testY)

# datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

it_train = datagen.flow(trainX, trainY, batch_size=batchSize)
steps = int(trainX.shape[0] / 64)
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_uniform', input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (2,2), padding='same', kernel_initializer='he_uniform' ,activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128,activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(0.4))

model.add(Dense(2,activation='softmax'))



# opt = SGD(lr=1e-3, momentum=0.9)
opt = Adam(lr=1e-3, decay=1e-3 / 20)
# opt = Adam(lr=1e-3, decay=0.9)

model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])


model.fit_generator(it_train, steps_per_epoch=len(trainX)//batchSize, epochs=20,
          validation_data=(testX,testY), verbose=1)

dir = './modelo/'
if not os.path.exists(dir):
    os.mkdir(dir)

model.save('./modelo/modelo.h5')
model.save_weights('./modelo/pesos.h5')