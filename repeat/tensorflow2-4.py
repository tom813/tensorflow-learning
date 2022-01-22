import tensorflow.keras as tfk
import numpy as np
import csv


def get_data(filename):
    with open(filename) as reading_file:
        csv_reader = csv.reader(reading_file)
        labels = []
        images = np.array([], dtype=float).reshape(0, 28, 28)
        for entry in csv_reader:
            if entry[0] == 'label':
                continue
            label_value = entry[0]
            pixel_values = np.array([float(i) for i in entry[1:785]])
            labels.append(float(label_value) - 1.0)
            pixel_values = pixel_values.reshape(1, 28, 28)
            images = np.concatenate((images, pixel_values))
        return labels, images


training_labels, training_images = get_data('../data/sign_mnist/sign_mnist_train.csv')
testing_labels, testing_images = get_data('../data/sign_mnist/sign_mnist_test.csv')

training_labels = tfk.utils.to_categorical(training_labels, num_classes=24, dtype=float)
testing_labels = tfk.utils.to_categorical(testing_labels, num_classes=24, dtype=float)

print(training_images.shape)
print(testing_images.shape)

training_images = np.reshape(training_images, (27455, 28, 28, 1))
testing_images = np.reshape(testing_images, (7172, 28, 28, 1))

model = tfk.Sequential([
    tfk.layers.Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=(28, 28, 1)),
    tfk.layers.MaxPool2D(2,2),
    tfk.layers.Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=(28, 28, 1)),
    tfk.layers.MaxPool2D(2,2),
    tfk.layers.Flatten(),
    tfk.layers.Dense(256, activation='relu'),
    tfk.layers.Dense(24, activation='softmax')
])

model.compile(
    optimizer=tfk.optimizers.RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    training_images,
    training_labels,
    epochs=20,
    validation_data=(testing_images, testing_labels)
)
