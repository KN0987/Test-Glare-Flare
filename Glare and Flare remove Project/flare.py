import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    # Decode without specifying the number of channels, it detects automatically
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])  # Resize to model expected input
    image /= 255.0  # Normalize to [0,1] range
    return image


def load_data(directory):
    images = []
    labels = []  # 1 for 'with_flare', 0 for 'without_flare'
    for label in ['with_flare', 'without_flare']:
        class_dir = os.path.join(directory, label)
        for img_path in os.listdir(class_dir):
            img = load_and_preprocess_image(os.path.join(class_dir, img_path))
            images.append(img)
            labels.append(1 if label == 'with_flare' else 0)
    return np.array(images), np.array(labels)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_images, train_labels = load_data(
    '/data/train')
test_images, test_labels = load_data(
    '/data/test')

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

model.fit(train_ds, epochs=15, validation_data=test_ds)

loss, accuracy = model.evaluate(test_ds)
print(f"Loss: {loss}, Accuracy: {accuracy}")
