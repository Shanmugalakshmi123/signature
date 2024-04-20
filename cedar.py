import pickle
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Define the paths to the real and forged signature images
real_path = r'C:\Users\shamr\Desktop\CEDAR\original'
forged_path = r'C:\Users\shamr\Desktop\CEDAR\forged'

# Load the real signature images
real_images = []
for filename in os.listdir(real_path):
    image = cv2.imread(os.path.join(real_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        real_images.append(image)

# Load the forged signature images
forged_images = []
for filename in os.listdir(forged_path):
    image = cv2.imread(os.path.join(forged_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        forged_images.append(image)

# Convert the images to numpy arrays
real_images = np.array(real_images)
forged_images = np.array(forged_images)

# Create the labels (0 for real, 1 for forged)
real_labels = np.zeros((real_images.shape[0], 1))
forged_labels = np.ones((forged_images.shape[0], 1))

# Concatenate the real and forged images and labels
images = np.concatenate((real_images, forged_images))
labels = np.concatenate((real_labels, forged_labels))

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
model.fit(train_images.reshape((-1, 224, 224, 1)), train_labels, batch_size=32, epochs=10, validation_data=(val_images.reshape((-1, 224, 224, 1)), val_labels))
#pickle.dump(model, open('cedar2.pkl', 'wb'),pickle.HIGHEST_PROTOCOL)
model.save('cedar3.h5')
# Evaluate the model on the validation data
loss, accuracy = model.evaluate(val_images.reshape((-1, 224, 224, 1)), val_labels)
print('Validation accuracy:', accuracy)
print('Model saved')