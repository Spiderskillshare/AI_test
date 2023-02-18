import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import urllib.request
import os

import tensorflow_datasets as tfds
# Check if the necessary files exist on disk, and if not, download them
if not os.path.exists('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'):
    print('Downloading InceptionV3 weights...')
    url = 'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    urllib.request.urlretrieve(url, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

# Add custom classification layers to the top of the model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers of the model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the dataset (here we are using the ImageNet dataset as a source of images)
(train_images, train_labels), (test_images, test_labels) = tfds.load('imagenet2012', split='train', shuffle_files=True)
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.imagenet.load_data()

# Load the ImageNet dataset
#dataset = tfds.load('imagenet2012', split='train', shuffle_files=True)
# Preprocess the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Train the model on the ImageNet dataset
model.fit(train_images, train_labels, epochs=10)

# Prompt the user to provide a photo and classify it as a table or no table
while True:
    # Ask the user for a photo
    filename = input("Enter the filename of a photo to classify (or 'quit' to exit): ")

    if filename == 'quit':
        break

    try:
        # Load and preprocess the photo
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Classify the photo
        prediction = model.predict(img)

        if prediction < 0.5:
            print("No table detected")
        else:
            print("Table detected")
    except:
        print("An error occurred while processing the photo. Please try again.")
