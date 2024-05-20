import pickle

# Load pre-trained SVM model and label encoder
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Get the unique classes from the SVM model
unique_classes = svm_model.classes_
num_classes = len(unique_classes)

print(f"Number of unique classes (faces) the SVM model has learned: {num_classes}")
print("Unique classes (labels):")
for i, class_label in enumerate(unique_classes):
    decoded_label = label_encoder.inverse_transform([class_label])[0]
    print(f"Class {i}: {decoded_label}")
