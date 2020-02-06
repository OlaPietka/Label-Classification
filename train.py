# python train.py -h

# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2  
import os 
import pickle  
import nn_arch 

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-e", "--epoch", required=True, help="number of epoch")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-a", "--augmentation", required=False, action='store_true', help="perform the augmentation")
args = vars(ap.parse_args())

# Collec the input data
# Batch size
BS = 32 

# Optionally resize images to the dimension of RESIZE x RESIZE. For LeNet5: 28
DO_RESIZE = True
RESIZE = 28 

# Images color depth (3 for RGB, 1 for grayscale)
IMG_DEPTH = 3
EPOCHS = int(args["epoch"])
DO_AUGMENTATION = args["augmentation"]
MODEL_BASENAME = args["model"]

# Initialize the data and labels
data = []  
labels = []  
labels_text = []

# Grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["dataset"])))

random.seed()
random.shuffle(image_paths)

# Loop over the input images
for image_path in image_paths:
    # Load the image and pre-process it
    image = cv2.imread(image_path)

    if DO_RESIZE:
        image = cv2.resize(image, (RESIZE, RESIZE))

    # Extract the class label (numeric and text) from the image path
    dir_name = image_path.split(os.path.sep)[-2]  
    dir_name_list = dir_name.split("-") 

    # Label is the number in the directory name, after first "-"
    label = dir_name_list[1]

    # Text label is the text in the directory name, after second "-"
    label_text = dir_name_list[2]

    # Store image and labels in lists
    data.append(image)
    labels.append(label)
    labels_text.append(label_text)

# Save the text labels to disk as pickle
classes = np.unique(labels_text)

f = open(MODEL_BASENAME + ".lbl", "wb") 
f.write((pickle.dumps(classes)))

# Convert labels to numpy array
labels = np.array(labels)

# Determine number of classes
no_classes = len(classes)

# Scale the raw pixel intensities to the [0, 1] range
data = np.array(data, dtype="float") / 255.0 

# Data partitioning (only if augmentation is enabled)
if DO_AUGMENTATION:
    # Partition the data into training and validating splits
    (train_data, valid_data, train_labels, valid_labels) = train_test_split(data, labels, test_size=0.2) 
else:
    # Data partitioning will be done automatically during training
    train_data = data
    train_labels = labels

# Convert the labels from integers to category vectors
train_labels = to_categorical(train_labels, num_classes=no_classes)

if DO_AUGMENTATION:
    valid_labels = to_categorical(valid_labels, num_classes=no_classes)

# Data augmentation
if DO_AUGMENTATION:
    # Construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

# Initialize the model
model = nn_arch.LeNet5.build(RESIZE, RESIZE, IMG_DEPTH, no_classes)

# Select the loss function
if no_classes == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
model.compile(loss=loss, optimizer="Adam", metrics=["accuracy"])

# Train the network
if DO_AUGMENTATION:
    H = model.fit_generator(aug.flow(train_data, train_labels, batch_size=BS), validation_data=(valid_data, valid_labels),
                  epochs=EPOCHS, verbose=1)
else:
    H = model.fit(x=train_data, y=train_labels, batch_size=BS, validation_split=0.2, epochs=EPOCHS, verbose=1)

# Save model to disk
model.save(MODEL_BASENAME + ".h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="train_loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(MODEL_BASENAME + ".png")