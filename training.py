import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

def extract_features(image):
    # Convert image to RGB if it is grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize image to the expected input size of the FaceNet model
    image = cv2.resize(image, (80, 80))
    
    # Normalize the image
    image = (image / 255.0 - 0.5) * 2.0
    
    # Convert to torch tensor
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    
    return features

# Load the pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

faces_dir = 'faces'
X = []
y = []

for person_name in os.listdir(faces_dir):
    person_dir = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, z, w, h) in faces:
            face_img = image[z:z+h, x:x+w]
            features = extract_features(face_img)
            break
        X.append(features)
        y.append(person_name)

X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X, y)

# Save the model and label encoder
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Get the unique classes from the SVM model
unique_classes = svm_model.classes_
num_classes = len(unique_classes)

print("Model Saved")
print(f"Unique Faces: {num_classes}")
for i, class_label in enumerate(unique_classes):
    decoded_label = label_encoder.inverse_transform([class_label])[0]
    print(f"Class {i}: {decoded_label}")
