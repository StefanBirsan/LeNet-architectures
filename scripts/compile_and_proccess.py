import matplotlib.pyplot as plt
from plot_utils import plot_training_history
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from custom_callbacks import TestAccuracyCallback
from sklearn.metrics import f1_score
from io import StringIO
            
def compile_train_evaluate_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, batch_size):
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Ensure the model's input layer matches the shape of X_train
    if model.input_shape[1:] != X_train.shape[1:]:
        raise ValueError(f"Model input shape {model.input_shape[1:]} does not match X_train shape {X_train.shape[1:]}")

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

    # Compilation of the model
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

    summary_io = StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
    
    # Ensure the directory exists
    os.makedirs("test_summaries", exist_ok=True)
    
    with open("test_summaries/model_ReLu_batch256_ep50_lr0.00001_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_io.getvalue())

    test_accuracy_callback = TestAccuracyCallback((X_test, y_test))

    # Training the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[test_accuracy_callback], batch_size=batch_size)

    # Plotting the training history
    #plot_training_history(history)

    return history, test_accuracy_callback.test_accuracy, test_accuracy_callback.test_loss

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
    
    # Make predictions on the test set
    y_pred = model.predict(preprocessed_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(true_class_ids, axis=1)

    # Calculate the F1-score
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Loss: {loss:.4f}')
    print(f'F1-score: {f1:.2f}')

    return loss, accuracy, f1