# Import the necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D

def lenet_5():
    model = Sequential()

    model.add(Conv2D(6, (5, 5), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(120, (5, 5), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(84))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model

def mvgg_5():
    model = Sequential()

    model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model

def mvgg_6():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model


def mvgg_7():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model


def mvgg_8():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model


def mvgg_9():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(Conv2D(48, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))

    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), padding='SAME'))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(4, 4), padding='SAME'))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model


