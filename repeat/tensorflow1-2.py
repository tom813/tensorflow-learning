import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

mnist = tfk.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255., x_test / 255.

model = tfk.Sequential([
    tfk.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=[28, 28, 1]),
    tfk.layers.MaxPool2D(2,2),
    tfk.layers.Flatten(),
    tfk.layers.Dense(512, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
