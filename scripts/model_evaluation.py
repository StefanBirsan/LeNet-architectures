import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.utils import to_categorical
from scripts.compile_and_proccess import compile_train_evaluate_plot, predict_images, preprocess_test_image
from scripts.plot_utils import plot_accuracy, plot_loss, plot_confusion_matrix, plot_roc_curve


def process_and_evaluate_model(model, annotations_path, base_path, X_train, y_train, X_val, y_val, epochs=50):
    # Process the test images
    annotations = pd.read_csv(annotations_path)
    image_paths = annotations['Path'].values
    true_class_ids = annotations['ClassId'].values
    true_class_ids = to_categorical(true_class_ids, num_classes=43)
    X_test = np.vstack([preprocess_test_image(path, base_path) for path in image_paths])
    y_test = true_class_ids

    # Compilation of the model
    history, test_accuracy, test_loss, confusion_matrix = compile_train_evaluate_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs)

    # Evaluate the model on the preprocessed images
    loss, accuracy, f1 = predict_images(model, annotations_path, base_path)

    # Plot results
    plot_accuracy(history, test_accuracy)
    plot_loss(history, test_loss)
    plot_confusion_matrix(confusion_matrix)

    # Calculate ROC curve and AUC score
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

    print(f'ROC AUC score: {roc_auc:.2f}')
    plot_roc_curve(fpr, tpr, roc_auc)

    return loss, accuracy, f1