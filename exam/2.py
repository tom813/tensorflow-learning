# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create and train a classifier for the MNIST dataset.
# Note that the test will expect it to classify 10 classes and that the 
# input shape should be the native size of the MNIST dataset which is 
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

def solution_model():
    mnist = tfk.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255., x_test / 255.
    print(y_train)
    #x_train = np.reshape(x_train, (60000, 28, 28, 1))
    #x_test = np.reshape(x_test, (10000, 28, 28, 1))
    print(x_test.shape)

    #exit()
    model = tfk.Sequential([
        tfk.layers.Conv2D(filters=1, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
        tfk.layers.MaxPool2D((2,2)),
        tfk.layers.Flatten(),
        tfk.layers.Dense(512, activation='relu'),
        tfk.layers.Dense(10, activation='softmax')
    ])
    #model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=100, verbose=1, validation_data=(x_test, y_test))
    # YOUR CODE HERE
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
