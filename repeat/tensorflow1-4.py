import tensorflow as tf
import tensorflow.keras as tfk
import numpy

datagen = tfk.preprocessing.image.ImageDataGenerator(
    rescale= 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
dataval = tfk.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    '../data/happy-or-sad-training',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=16
)

validation_generator = dataval.flow_from_directory(
    '../data/happy-or-sad-validation',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=16
)

model = tfk.Sequential([
    tfk.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=[150, 150, 3]),
    tfk.layers.MaxPool2D(3,3),
    tfk.layers.Conv2D(filters=32, kernel_size=(3,3)),
    tfk.layers.MaxPool2D(3,3),
    tfk.layers.Conv2D(filters=32, kernel_size=(3,3)),
    tfk.layers.MaxPool2D(3,3),
    tfk.layers.Flatten(),
    tfk.layers.Dense(512, activation='relu'),
    tfk.layers.Dense(64, activation='relu'),
    tfk.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tfk.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=50, validation_data=validation_generator)