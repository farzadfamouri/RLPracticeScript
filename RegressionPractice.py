import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import urllib.request

url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv"
urllib.request.urlretrieve(url, "moore.csv")

data = pd.read_csv("moore.csv", header=None).to_numpy()
#print(data.head())

X = data[:, 0].reshape(-1, 1) #making an NXD
Y = data[:, 1]

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

X = X - X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='mse', )

def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001

schedule = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, Y, epochs=200, callbacks=[schedule])

# Plot the loss
plt.plot(r.history['loss'], label='loss')

plt.legend()

# Get the slope of the line
# The slope of the line is related to the doubling rate of transistor count
print(model.layers) # Note: there is only 1 layer, the "Input" layer doesn't count
print(model.layers[0].get_weights())

# The slope of the line is:
a = model.layers[0].get_weights()[0][0,0]