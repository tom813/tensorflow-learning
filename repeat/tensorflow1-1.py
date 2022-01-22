import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model = tfk.Sequential([
    tfk.layers.Dense(1, input_shape=[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(xs, ys, epochs=8000)
print(model.predict([2.0]))