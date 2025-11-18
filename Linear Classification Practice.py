import tensorflow as tf
from IPython.testing.decorators import skipif
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(type(data))
print("data: ", data.data)
print('keys: ', data.keys())
print('Shape: ',data.data.shape)
print("target: ", data.target)
print("target name: ", data.target_names)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( data.data, data.target, test_size = 0.33, random_state = 42)

N, D = x_train.shape
print(N)
print(D)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32)

print("Train Score: ", model.evaluate(x_train, y_train))
print("Test Score: ", model.evaluate(x_test, y_test))

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label ='loss')
plt.plot(r.history['val_loss'], label ='val_loss')
plt.legend()