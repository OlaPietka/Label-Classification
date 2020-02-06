# Import packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import pickle

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-t", "--testset", required=True, help="path to test images")
args = vars(ap.parse_args())

# Resize parmeters (RESIZE should be the same as used in training)
DO_RESIE = True
RESIZE = 28

# Read labels for classes to recognize
f = open(args["model"] + ".lbl", "rb")
CLASS_LABELS = pickle.load(f)
f.close()

# Load the trained network
model = load_model(args["model"] + ".h5")
model.summary()

# Loop over images
for image_name in os.listdir(args["testset"]):
    # Load the image
    image = cv2.imread(args["testset"] + os.path.sep + image_name)
    output = imutils.resize(image, width=400)

    # Preprocess the image
    if DO_RESIE:
        image = cv2.resize(image, (RESIZE, RESIZE))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Classify the input image
    predict = model.predict(image)[0]

    # Find the winner class and the probability
    probability = predict * 100
    winners_indexes = np.argsort(probability)[::-1]

    # Build the label
    for (i, index) in enumerate(winners_indexes):
        label = "{}: {:.2f}%".format(CLASS_LABELS[index], probability[index])

        # Draw the label on the image
        cv2.putText(output, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
