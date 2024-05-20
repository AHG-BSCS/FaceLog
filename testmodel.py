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
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    features = hog.compute(image).flatten()
    return features

# Load the specific test image
test_image_path = 'faces/Jhondale/02.jpg'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(test_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    face_img = test_image[y:y+h, x:x+w]
    resized_test_image = cv2.resize(face_img, (64, 64))
    test_features = extract_features(resized_test_image)
    prediction = svm_model.predict([test_features])
    proba = svm_model.predict_proba([test_features]).max()
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    print(f"{predicted_label} : {proba}%")
    break
