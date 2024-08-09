import numpy as np
import random
import os
import sys
import time
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from networks.lenet_1 import LeNet
from scripts.model_evaluation import process_and_evaluate_model

def main():
    images = []
    labels = []
    classes = 43

    train_path = r'.\datasets\GTSRB_dataset\Train'
    base_path = r'.\datasets\GTSRB_dataset'
    annotations_path = r'.\datasets\GTSRB_dataset\Test.csv'

    for i in range(classes):
        path = os.path.join(train_path, str(i))
        img_folder = os.listdir(path)
        for j in img_folder:
            try:
                image = cv.imread(os.path.join(path, j))
                image = cv.resize(image, (28, 28))
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = np.array(image)
                images.append(image)
                label = np.zeros(classes)
                label[i] = 1.0
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {j}: {e}")

    # Normalize images and convert labels to numpy arrays
    images = np.array(images) / 255
    labels = np.array(labels)
    print('Images shape:', images.shape)
    print('Labels shape:', labels.shape)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images.astype(np.float32), labels.astype(np.float32), test_size=0.2, random_state=123)

    # Build the model
    model = LeNet.build(numChannels=3, imgRows=28, imgCols=28, numClasses=43, activation='relu')

    start_time = time.time()

    # Process and evaluate the model
    loss, accuracy, f1, recall = process_and_evaluate_model(model, annotations_path, base_path, X_train, y_train, X_val, y_val, epochs=50, batch_size=256)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    print(f'Test F1 Score: {f1}')
    print(f'Test Recall: {recall}')

    # Calculate precision
    y_pred = model.predict(X_val)
    y_val_labels = np.argmax(y_val, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    precision = precision_score(y_val_labels, y_pred_labels, average='weighted')
    print(f'Precision: {precision}')


if __name__ == "__main__":
    main()