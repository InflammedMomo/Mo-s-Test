import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt


n=500
train = tf.keras.preprocessing.image_dataset_from_directory('Project 2 Data/Data/train',
                                                           color_mode = 'rgb',
                                                            image_size=(n, n),
                                                           batch_size = 32)
test = tf.keras.preprocessing.image_dataset_from_directory('Project 2 Data/Data/test',
                                                           color_mode = 'rgb',
                                                           image_size=(n, n),
                                                           batch_size = 32)
valid = tf.keras.preprocessing.image_dataset_from_directory('Project 2 Data/Data/valid',
                                                           color_mode = 'rgb',
                                                            image_size=(n, n),
                                                           batch_size = 32)
targets = ['crack','missing head','paint off']
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(n, n, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3,seed=42))
model.add(layers.Dense(128, activation='elu'))
model.add(layers.Dropout(0.4,seed=42))
model.add(layers.Dense(4, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train, epochs=20, validation_data=valid,batch_size=32)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(history.history['accuracy'], label='accuracy')
ax1.plot(history.history['val_accuracy'], label = 'val_accuracy')
ax1.set(ylabel='Accuracy')
ax1.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test,  verbose=2)
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label = 'Val_loss')
ax2.set(ylabel='Loss',xlabel='Epoch')
ax2.legend(loc='lower right')
plt.show()

model.save('model3.keras')
