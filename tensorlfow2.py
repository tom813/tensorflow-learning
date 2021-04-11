import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training")


#loads testing and training images 28 * 28 images with color values from 0 to 255
mnnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnnist.load_data()

#plt.imshow(training_images[0])
#print(training_labels[0])
#print(training_images[0])

#It's better to train with values between 0 and 1
training_images = training_images / 255.0
test_images = test_images / 255.0

#Sequence defines the neural network I think
#Flatten turns the matrix (28 x 28) into a vektor with 1 x 784
#activation should be a normal sigmoid function
#relu is a activation function like sigmoid
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classification = model.predict(test_images)
#probabilities
print(classification[0])
#in cause of softmax the sum of the values is 1
print(np.sum(classification[0]))