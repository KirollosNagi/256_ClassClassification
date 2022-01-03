from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.utils import np_utils

# Path to folders with training data
test_path = Path("test_data")

images = []
labels = []

# Load all the training images
list_dir = [d for d in test_path.iterdir() if d.is_dir()]

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
x_test = np.array(images)

# Also convert the labels to a numpy array
labels = np_utils.to_categorical(labels, 257)
y_test = np.array(labels)
print(y_test.shape)

# Save the array of extracted features to a file
joblib.dump(x_test, "x_test.dat")

# Save the matching array of expected values to a file
joblib.dump(y_test, "y_test.dat")
