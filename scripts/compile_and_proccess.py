import matplotlib.pyplot as plt
from plot_utils import plot_training_history
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from custom_callbacks import TestAccuracyCallback

            
def compile_train_evaluate_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs):
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Ensure the model's input layer matches the shape of X_train
    if model.input_shape[1:] != X_train.shape[1:]:
        raise ValueError(f"Model input shape {model.input_shape[1:]} does not match X_train shape {X_train.shape[1:]}")

    # Compilation of the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()

    test_accuracy_callback = TestAccuracyCallback((X_test, y_test))

    # Training the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[test_accuracy_callback])

    # Plotting the training history
    #plot_training_history(history)

    return history, test_accuracy_callback.test_accuracy, test_accuracy_callback.test_loss, test_accuracy_callback.confusion_matrix

def preprocess_test_image(image_relative_path, base_image_path):
    normalized_path = os.path.normpath(image_relative_path)
    full_path = os.path.join(base_image_path, normalized_path)
    img = load_img(full_path, target_size=(28, 28))  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  
    return img

def predict_images(model, annotations_path, base_image_path):
    annotations = pd.read_csv(annotations_path)
    image_paths = annotations['Path'].values
    true_class_ids = annotations['ClassId'].values
    true_class_ids = to_categorical(true_class_ids, num_classes=43)

    preprocessed_images = np.vstack([preprocess_test_image(path, base_image_path) for path in image_paths])
    
    # Evaluate the model on the preprocessed images
    loss, accuracy = model.evaluate(preprocessed_images, true_class_ids, verbose=0)
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Loss: {loss:.4f}')

    return loss, accuracy