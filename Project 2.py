import tensorflow as tf
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
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
targets = ['crack','missing head','paint off']
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(10))
#model.add(layers.Dropout(0,seed=42))

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=20, validation_data=valid)




plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.clf()

test_loss, test_acc = model.evaluate(test,  verbose=2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()
