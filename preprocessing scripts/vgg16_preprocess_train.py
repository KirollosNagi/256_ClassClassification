from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
from keras.utils import np_utils

# Path to folders with training data
training_path = Path("training_data")

images = []
labels = []

# Load all the training images
list_dir = [d for d in training_path.iterdir() if d.is_dir()]

for d in list_dir:
    for img in d.glob("*.jpg"):
        # Load the image from disk
        size = (64, 64)
        img = image.load_img(img, target_size=size)

        # Convert the image to a numpy array
        image_array = image.img_to_array(img)

        # Add the image to the list of images
        images.append(image_array)

        # For each class, the expected value should be the class number
        labels.append(int(str(d).partition('\\')[2].partition('.')[0])-1)

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
labels = np_utils.to_categorical(labels, 257)
y_train = np.array(labels)
print(y_train.shape)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")
