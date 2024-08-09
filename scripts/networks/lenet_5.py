from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout, Input, MaxPooling2D

class LeNet_5:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation):
              model = Sequential()

              # Define the input shape
              inputShape = (imgRows, imgCols, numChannels)

              # Adding the layers to the model
              model.add(Input(shape=inputShape))
              model.add(Conv2D(filters=4, kernel_size=(5, 5), activation=activation))
              model.add(AveragePooling2D(pool_size=(2, 2)))
              model.add(Conv2D(filters=12, kernel_size=(5, 5), activation=activation))
              model.add(AveragePooling2D(pool_size=(2, 2)))
              model.add(Flatten())
              model.add(Dense(units=numClasses, activation='softmax'))

              return model 