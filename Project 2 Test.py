from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
n=500

test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    'Project 2 Data/Data/test',
    color_mode='rgb',
    target_size=(n, n),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


model = load_model('model1.keras')
image_paths = ["Project 2 Data/Data/test/crack/test_crack.jpg",
               "Project 2 Data/Data/test/missing-head/test_missinghead.jpg",
               "Project 2 Data/Data/test/paint-off/test_paintoff.jpg"]


for image_path in image_paths:
    img1=load_img(image_path, target_size=(n, n),color_mode='grayscale')
    img = img_to_array(img1)

    img/= 255.
    img=np.expand_dims(img, axis=0)
    print(img.ndim)
    pred=model.predict(img)

    class_index = np.argmax(pred, axis=1)[0]
    index_to_class = {v: k for k, v in test_generator.class_indices.items()}
    predicted_class = index_to_class[class_index]
    print(predicted_class)

    plt.figure()

    plt.imshow(img1)
    plt.title(f'Predicted class: {predicted_class}')

    pred_percentages = pred[0] * 100
    pred_percentages = np.round(pred_percentages, 2)
    pred_percentages_str = ', '.join(map(str, pred_percentages))
    plt.figtext(0.5, 0.01, f'Predicted probabilities (cracked, missing,paint off): ({pred_percentages_str})%', wrap=True,horizontalalignment='center', fontsize=12)
    plt.show()