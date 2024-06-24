import numpy as np # MATRIX OPERATIONS
import pandas as pd # EFFICIENT DATA STRUCTURES
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import math # MATHEMATICAL OPERATIONS
import cv2 # IMAGE PROCESSING - OPENCV
from glob import glob # FILE OPERATIONS
import itertools
#ali
import joblib

# KERAS AND SKLEARN MODULES
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# GLOBAL VARIABLES
scale = 160 #70
seed = 90

path_to_images = 'wm-dataset/dataset-resized_new/*/*.jpg'
images = glob(path_to_images)
trainingset = []
traininglabels = []
num = len(images)
count = 1
#READING IMAGES AND RESIZING THEM
for i in images:
    print(str(count)+'/'+str(num),end='\r')
    trainingset.append(cv2.resize(cv2.imread(i),(scale,scale)))
    traininglabels.append(i.split('\\')[-2])
    count=count+1
trainingset = np.asarray(trainingset)
# Save the label encoder to a file

traininglabels = pd.DataFrame(traininglabels)

# Encode labels
labels = preprocessing.LabelEncoder()
labels.fit(traininglabels[0])
print('Classes' + str(labels.classes_))
encodedlabels = labels.transform(traininglabels[0])
clearalllabels = np_utils.to_categorical(encodedlabels)
classes = clearalllabels.shape[1]
print(str(classes))

# Save the label encoder
joblib.dump(labels, 'Ali_2.pkl')

# Display class distribution
traininglabels[0].value_counts().plot(kind='pie')

# Normalize the data
new_train = trainingset / 255.0
x_train, x_test, y_train, y_test = train_test_split(new_train, clearalllabels, test_size=0.1, random_state=seed, stratify=clearalllabels)

# Data augmentation
generator = ImageDataGenerator(rotation_range=180, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)
generator.fit(x_train)

# Model definition
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Training the model
model.fit(generator.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=50, verbose=1)

# Evaluate the model
print(model.evaluate(x_train, y_train))  # Evaluate on train set
print(model.evaluate(x_test, y_test))  # Evaluate on test set

# Confusion matrix
y_pred = model.predict(x_test)
y_class = np.argmax(y_pred, axis=1)
y_check = np.argmax(y_test, axis=1)
cmatrix = confusion_matrix(y_check, y_class)
print(cmatrix)

# Display test images with predictions
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_test[i])
    print("Predicted: " + labels.classes_[y_class[i]])
    print("Actual class: " + labels.classes_[y_check[i]])

# Save the model
model.save('Ali_2.h5')