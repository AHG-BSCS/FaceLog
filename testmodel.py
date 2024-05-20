import cv2
import numpy as np
import pickle

# Load pre-trained SVM model and label encoder
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to extract HOG features
def extract_features(image):
    winSize = (256, 256)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    features = hog.compute(image).flatten()
    return features

# Load the specific test image
test_image_path = 'faces/jhondale/01.jpg'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image: resize to 64x64 (or the size used during training)
resized_test_image = cv2.resize(test_image, (128, 128))

# Extract HOG features from the test image
test_features = extract_features(resized_test_image)

# Predict using the SVM model
prediction = svm_model.predict([test_features])
predicted_index = prediction[0]

# Decode the prediction to get the class label
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

print(f"The predicted label for the test image is: {predicted_label}")
