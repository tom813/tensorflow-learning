import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    '../data/kagglecatsanddogs_3367a/PetImages',
    target_size=(150, 150),
    class_mode='binary',
    subset="training",
    batch_size=16
)

validation_generator = train_datagen.flow_from_directory(
    '../data/kagglecatsanddogs_3367a/PetImages',
    target_size=(150, 150),
    class_mode='binary',
    subset="validation",
    batch_size=16
)

model = tfk.Sequential([
    tfk.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
    tfk.layers.MaxPool2D(2,2),
    tfk.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tfk.layers.MaxPool2D(2,2),
    tfk.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tfk.layers.MaxPool2D(2,2),
    tfk.layers.Flatten(),
    tfk.layers.Dense(256, activation='relu'),
    tfk.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)
