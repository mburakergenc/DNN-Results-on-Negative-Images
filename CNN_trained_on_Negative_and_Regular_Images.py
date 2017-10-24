# Import the necessary libraries
import numpy as np
from PIL import Image
import glob
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

def run_model(model):
    # Load the MNIST data provided by keras
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reconstruct the Training images with the negative ones
    # Turn them into numpy arrays
    training_images = "mnist/train-images/*.jpg"
    test_images = "mnist/test-images/*.jpg"
    train_image_list = glob.glob(training_images)
    test_image_list = glob.glob(test_images)

    X_neg_train =  np.array([np.array(Image.open(img)) for img in train_image_list])
    X_neg_test =  np.array([np.array(Image.open(img)) for img in test_image_list])

    # Concatenate the Regular and Negative Image NP Arrays
    X_train_f = np.concatenate((X_train, X_neg_train),axis=0)
    X_test_f = np.concatenate((X_test, X_neg_test), axis=0)
    y_train_f = np.concatenate((y_train, y_train), axis=0)
    y_test_f = np.concatenate((y_test, y_test), axis=0)

    # The shape of each array is (60000, 28, 28)
    # Since we are using tensorflow the format of each array should be (batch, height, width, channels)
    # Let's reshape it then!
    X_train_f = X_train_f.reshape(X_train_f.shape[0], 28, 28, 1)
    X_test_f = X_test_f.reshape(X_test_f.shape[0], 28, 28, 1)

    X_train_f = X_train_f.astype('float32')
    X_test_f = X_test_f.astype('float32')

    X_train_f/=255
    X_test_f/=255

    # Shuffle the data (can also be done with the fit function. Approximately gives the same result)
    indexes = np.random.permutation(len(X_train_f))
    X_train_f,y_train_f = X_train_f[indexes], y_train_f[indexes]

    # Apply one-hot encoding
    number_of_classes = 10
    y_train_f = np_utils.to_categorical(y_train_f, number_of_classes)
    y_test_f = np_utils.to_categorical(y_test_f, number_of_classes)

    # Now let's create the CNN model that will classify the MNIST images




    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Apply data augmentation to reduce over-fitting
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08,
                             zoom_range=0.08)

    test_gen = ImageDataGenerator()

    # Create batches in order to use less memory.
    # Using batch of 64, the model will take 64 images at a time in the process of training
    train_generator = gen.flow(X_train_f, y_train_f, batch_size=64)
    test_generator = test_gen.flow(X_test_f, y_test_f, batch_size=64)

    # We're ready to train the model
    # shuffle argument can be used here as well
    model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5,
                        validation_data=test_generator, validation_steps=10000//64)

