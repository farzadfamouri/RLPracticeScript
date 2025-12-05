import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""
import urllib.request

url = "https://archive.org/download/food-5-k/Food-5K.zip"
urllib.request.urlretrieve(url, "Food-5K.zip")

print("Download complete!")
"""


"""
import zipfile

with zipfile.ZipFile("Food-5K.zip", "r") as zip_ref:
    zip_ref.extractall("Food-5K")
"""


"""
# look at an image for fun
plt.imshow(image.load_img('training/0_808.jpg'))
plt.show()"""


import os
import shutil


"""
# Create directories
os.makedirs("data/train/nonfood", exist_ok=True)
os.makedirs("data/train/food", exist_ok=True)
os.makedirs("data/test/nonfood", exist_ok=True)
os.makedirs("data/test/food", exist_ok=True)

# Move images
for src, dst in [
    ("training/0", "data/train/nonfood"),
    ("training/1", "data/train/food"),
    ("validation/0", "data/test/nonfood"),
    ("validation/1", "data/test/food"),
]:
    for file in os.listdir(os.path.dirname(src)):
        if file.startswith(os.path.basename(src)) and file.endswith(".jpg"):
            shutil.move(os.path.join(os.path.dirname(src), file), dst)

"""

train_path = 'data/train'
valid_path = 'data/test'
# These images are pretty big and of different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [200, 200]
# useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')
# useful for getting number of classes
folders = glob(train_path + '/*')

print("Folders Name: ", folders)

K = len(folders)
print("Length of folders: ", K)

"""
# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()
"""

ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False)

# freeze pretrained model weights
ptm.trainable = False

# data augmentation
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)

from tensorflow.keras.layers import Input


# Build the model using the functional API
i = Input(shape=IMAGE_SIZE + [3])
x = preprocess_input(i)
x = data_augmentation(x)
x = ptm(x)
x = Flatten()(x)
x = Dense(K, activation='softmax')(x)
model = Model(inputs=i, outputs=x)

model.summary()

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

batch_size = 128

# --- create raw datasets FIRST ---
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
  train_path,
  image_size=IMAGE_SIZE,
  batch_size=batch_size
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
  valid_path,
  image_size=IMAGE_SIZE,
  batch_size=batch_size
)

# --- get class names BEFORE wrapping (fix) ---
class_names = raw_train_ds.class_names
K = len(class_names)
print(class_names)

# --- now wrap for performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# fit the model
r = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()