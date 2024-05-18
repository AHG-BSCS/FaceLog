import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

def load_images_from_folder(folder):
    images = []
    labels = []
    label_names = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            label_names.append(label)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use grayscale for simplicity
                if image is not None:
                    images.append(image)
                    labels.append(label)
    return images, labels, label_names

def extract_features(images):
    features = []
    hog = cv2.HOGDescriptor()
    for image in images:
        # Resize image to a fixed size (e.g., 64x64) for consistency
        image = cv2.resize(image, (64, 64))
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        # Extract HOG features
        feature = hog.compute(image).flatten()
        features.append(feature)
    return features

# Load images and labels
folder_path = 'faces'
images, labels, label_names = load_images_from_folder(folder_path)

# Extract features from images
features = extract_features(images)

# Encode labels into numeric values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels_encoded)

# Train the SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(X, y)

# Save the trained model and the label encoder
model_path = 'models/svm_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(svm, f)

label_encoder_path = 'models/label_encoder.pkl'
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved successfully.")
