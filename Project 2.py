import tensorflow as tf
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image
train = tf.keras.preprocessing.image_dataset_from_directory('Project 2 Data/Data/train',
                                                           color_mode = 'rgb',
                                                            image_size=(50, 50),
                                                           batch_size = 32)
test = tf.keras.preprocessing.image_dataset_from_directory('Project 2 Data/Data/test',
                                                           color_mode = 'rgb',
                                                           image_size=(50, 50),
                                                           batch_size = 32)
valid = tf.keras.preprocessing.image_dataset_from_directory('Project 2 Data/Data/valid',
                                                           color_mode = 'rgb',
                                                            image_size=(50, 50),
                                                           batch_size = 32)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()