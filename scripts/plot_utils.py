import matplotlib.pyplot as plt
from keras.callbacks import Callback


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