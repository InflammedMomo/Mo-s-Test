import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator


n=500

datagen = ImageDataGenerator(
    zoom_range= 0.2,
    rotation_range = 15,
    horizontal_flip = True,
    rescale = 1./255
)
train = datagen.flow_from_directory(
    'Project 2 Data/Data/train',
    target_size = (n, n),
    color_mode = 'grayscale',
    batch_size = 32,
    shuffle = True,
    class_mode = 'categorical')

valid= datagen.flow_from_directory(
    'Project 2 Data/Data/valid',
    target_size = (n, n),
    color_mode = 'grayscale',
    batch_size = 32,
    shuffle = True,
    class_mode = 'categorical')


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(n, n,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dropout(0.4,seed=42))
model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dropout(0.3,seed=42))
model.add(layers.Dense(3, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train, epochs=20, validation_data=valid)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(history.history['accuracy'], label='accuracy')
ax1.plot(history.history['val_accuracy'], label = 'val_accuracy')
ax1.set(ylabel='Accuracy')
ax1.legend(loc='lower right')


ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label = 'Val_loss')
ax2.set(ylabel='Loss',xlabel='Epoch')
ax2.legend(loc='lower right')
plt.show()

model.save('model1.keras')
