import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.test_accuracy = []
        self.test_loss = []
        self.confusion_matrix = None

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.test_data
        # Evaluate the model on the test data
        loss, accuracy = self.model.evaluate(x, y, verbose=0)
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)

        # Predict the classes
        y_pred = np.argmax(self.model.predict(x), axis=1)
        y_true = np.argmax(y, axis=1)
        
        # Compute confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)