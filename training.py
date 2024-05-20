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
    image_paths = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            label_names.append(label)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(label)
                    image_paths.append(image_path)
    return images, labels, label_names, image_paths

def extract_features_and_clean(images, image_paths):
    features = []
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    for image, image_path in zip(images, image_paths):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            os.remove(image_path)
            print(f"Face deleted: {image_path}")
        else:
            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                newimage = cv2.resize(face_img, (64, 64))
                feature = hog.compute(newimage).flatten()
                features.append(feature)
                break
    return features

# Load images and labels
folder_path = 'faces'
images, labels, label_names, image_paths = load_images_from_folder(folder_path)

# Extract features from images and delete images without faces
features = extract_features_and_clean(images, image_paths)

# Encode labels into numeric values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train the SVM model
X = np.array(features)
y = np.array(labels_encoded)
svm = SVC(kernel='linear', probability=True)
svm.fit(X, y)

model_path = 'models/svm_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(svm, f)

label_encoder_path = 'models/label_encoder.pkl'
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

# Get the unique classes from the SVM model
unique_classes = svm.classes_
num_classes = len(unique_classes)

print("Model Saved")
print(f"Unique Faces: {num_classes}")
for i, class_label in enumerate(unique_classes):
    decoded_label = label_encoder.inverse_transform([class_label])[0]
    print(f"Class {i}: {decoded_label}")
