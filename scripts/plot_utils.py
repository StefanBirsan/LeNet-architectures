import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import os


def plot_training_history(history):
    plt.figure(0)
    plt.title('Learning Curves')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1])
    plt.legend(loc='lower right')

    plt.figure(1)
    plt.title('Loss Curves')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 0.2])
    plt.legend(loc='upper right')

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/training_history.png")

    plt.show()

def plot_accuracy(history, test_accuracy):
    # Plot testingg & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['val_accuracy'])
    plt.plot(test_accuracy, label='test_accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Validation', 'Test'], loc='upper left')

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/model_accuracy.png")

    plt.show()

def plot_loss(history, test_loss):
    # Plot testing & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['val_loss'])
    plt.plot(test_loss, label='test_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Validation', 'Test'], loc='upper left')

    os.makedirs("plots", exist_ok=True)
    
    # Save the plot
    plt.savefig("plots/model_loss.png")

    plt.show()

def plot_confusion_matrix(confusion_matrix, classes):
    conf_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10)) 
    ax = plt.gca()
    
    disp = ConfusionMatrixDisplay(conf_matrix_normalized, display_labels=range(classes))
    disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)
    
    # Adjust font sizes
    plt.xticks(fontsize=8, rotation=45)  
    plt.yticks(fontsize=8)  
    for text in disp.text_.ravel():
        text.set_fontsize(6)  # Decrease font size for annotations
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.gca().images[-1].colorbar.ax.tick_params(labelsize=10)  
    plt.title('Confusion Matrix', fontsize=14) 
    plt.xlabel('Predicted label', fontsize=12)  
    plt.ylabel('True label', fontsize=12)  

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/confusion_matrix.png")

    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    os.makedirs("plots", exist_ok=True)
    
    # Save the plot
    plt.savefig("plots/roc_curve.png")

    plt.show()    