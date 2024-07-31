import numpy as np
import random
import os
import sys
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout, Input, MaxPooling2D

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from plot_utils import plot_accuracy, plot_loss, plot_confusion_matrix
from compile_and_proccess import compile_train_evaluate_plot, preprocess_test_image, predict_images
from networks.lenet_1 import LeNet

images = []
labels = []
classes = 43

train_path = train_path = r'.\datasets\GTSRB_dataset\Train'
testing_path = r'.\datasets\GTSRB_dataset\Test'
base_path = r'.\datasets\GTSRB_dataset'
annotations_path = r'.\datasets\GTSRB_dataset\Test.csv'

for i in range(classes):
    path = os.path.join(train_path, str(str(i)))
    img_folder = os.listdir(path)
    for j in img_folder:
        try:
           image = cv.imread(os.path.join(path, j))
           image = cv.resize(image, (28,28))
           image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
           image = np.array(image)
           images.append(image)
           label = np.zeros(classes)
           label[i] = 1.0
           labels.append(label)
        except:
            pass

images = np.array(images)
images = images/255
labels = np.array(labels)
print('Images shape:', images.shape)
print('Labels shape:', labels.shape)

X = images.astype(np.float32)
y = labels.astype(np.float32)
# Random state ensures that the splits that you generate are reproducible
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)


#Building the model using the LeNet class

model = LeNet.build(numChannels=3, imgRows=28, imgCols=28, numClasses=43, activation='sigmoid')

# Process the test images
annotations = pd.read_csv(annotations_path)
image_paths = annotations['Path'].values
true_class_ids = annotations['ClassId'].values
true_class_ids = to_categorical(true_class_ids, num_classes=43)
X_test = np.vstack([preprocess_test_image(path, base_path) for path in image_paths])
y_test = true_class_ids

# Compilation of the model
history, test_accuracy, test_loss, confusion_matrix = compile_train_evaluate_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=50)

# Evaluate the model on the preprocessed images
loss, accuracy = predict_images(model, annotations_path, base_path)

plot_accuracy(history, test_accuracy)
plot_loss(history, test_loss)
plot_confusion_matrix(confusion_matrix)

