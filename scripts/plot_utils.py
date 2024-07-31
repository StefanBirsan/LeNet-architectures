import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import ConfusionMatrixDisplay


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
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()  
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap=plt.cm.Blues, ax=ax) 
    plt.title('Confusion Matrix')
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
    plt.show()    