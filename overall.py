import pickle
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Define the paths to the real and forged signature images
cedar_real_path = r'C:\Users\shamr\Desktop\CEDAR\original'
cedar_forged_path = r'C:\Users\shamr\Desktop\CEDAR\forged'
bengali_real_path = r'C:\Users\shamr\Desktop\sign1'
bengali_forged_path = r'C:\Users\shamr\Desktop\signf'
hindhi_real_path = r'C:\Users\shamr\Desktop\Hindhi\original'
hindhi_forged_path = r'C:\Users\shamr\Desktop\Hindhi\forged'

# Load the real signature images
real_images = []
for filename in os.listdir(cedar_real_path):
    image = cv2.imread(os.path.join(cedar_real_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        real_images.append(image)
for filename in os.listdir(bengali_real_path):
    image = cv2.imread(os.path.join(bengali_real_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        real_images.append(image)
for filename in os.listdir(hindhi_real_path):
    image = cv2.imread(os.path.join(hindhi_real_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        real_images.append(image)

# Load the forged signature images
forged_images = []
for filename in os.listdir(cedar_forged_path):
    image = cv2.imread(os.path.join(cedar_forged_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        forged_images.append(image)
for filename in os.listdir(bengali_forged_path):
    image = cv2.imread(os.path.join(bengali_forged_path, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        forged_images.append(image)
for filename in os.listdir(hindhi_forged_path):
    image = cv2.imread(os.path.join(hindhi_forged_path, filename))
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

from keras import backend as K

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'f1_score', 'precision', 'recall'])
# Compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
model.fit(train_images.reshape((-1, 224, 224, 1)), train_labels, batch_size=32, epochs=10, validation_data=(val_images.reshape((-1, 224, 224, 1)), val_labels))
#pickle.dump(model, open('cedar2.pkl', 'wb'),pickle.HIGHEST_PROTOCOL)
model.save('overall3.h5')
# Evaluate the model on the validation data


# fit the model
#history = model.fit(train_images, train_labels, validation_split=0.3, epochs=10, verbose=0)

# evaluate the model
#loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)
loss, accuracy, f1_score, precision, recall = model.evaluate(val_images.reshape((-1, 224, 224, 1)), val_labels)
print('Validation accuracy:', accuracy)
print("f1_score: ", f1_score)
print("precision: ", precision)
print("recall: ", recall)


print('Model saved')