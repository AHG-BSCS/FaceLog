import os
import cv2
import time
import pickle
import torch
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from threading import Thread
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import datetime

# Load pre-trained SVM model and label encoder
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

app = Flask(__name__)
camera = None
camera_mode = None
camera_index = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_recognition', methods=['GET'])
def face_recognition():
    global camera
    global camera_mode
    # global camera_index

    # camera_index = int(request.args.get('cameraIndex'))
    # camera_index -= 1
    if camera is None:
        camera = VideoCamera()
        camera_mode = 'recognition'
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_capturing', methods=['GET'])
def face_capturing():
    global camera
    global camera_mode
    # global camera_index
    
    # camera_index = int(request.args.get('cameraIndex'))
    # camera_index -= 1
    if camera is None:
        camera = VideoCamera()
        camera_mode = 'capture'
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while camera.running:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/stop_feed')
def stop_feed():
    global camera
    global camera_mode

    if camera is not None:
        camera.__del__()
        camera = None
        camera_mode = None
    return 'Webcam stopped'

@app.route('/capture_images', methods=['POST'])
def capture_images():
    global camera
    global camera_mode

    user_name = request.json.get('user_name')
    if not user_name:
        return jsonify({"error": "Username is required"}), 400
    
    user_folder = os.path.join('faces', user_name)
    os.makedirs(user_folder, exist_ok=True)

    # Capture images
    count = 0
    while count < 50:
        if camera is None:
            break
        frame = camera.capture_frame()
        file_path = os.path.join(user_folder, f'{count + 1}.jpg')
        cv2.imwrite(file_path, frame)
        count += 1
        cv2.waitKey(800)

    return jsonify({"message": f"Captured {count} images"}), 200

@app.route('/training')
def training():
    global svm_model
    global label_encoder
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
    return jsonify({"message" : f"Registered Faces: {num_classes}"})
    
# @app.route('/list_cameras', methods=['GET'])]
# def get_cameras():
#     cameras = list_cameras()
#     return jsonify(cameras)

# Threading for video capture and processing
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.grabbed, self.frame = self.video.read()
        self.last_recognition_time = time.time()
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            self.grabbed, self.frame = self.video.read()

    def get_frame(self):
        frame = self.frame.copy()
        current_time = time.time()

        # Perform face recognition every 0.1 seconds
        if current_time - self.last_recognition_time >= 0.1:
            if camera_mode == 'recognition':
                frame = recognize_faces(frame)
                self.last_recognition_time = current_time
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def capture_frame(self):
        return self.frame.copy()
    
    def __del__(self):
        self.running = False
        self.video.release()

def recognize_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a DataFrame to store the detection results
    if not os.path.exists('attendance.xlsx'):
        df = pd.DataFrame(columns=['Person', 'Date', 'Time', 'Probability'])
    else:
        df = pd.read_excel('attendance.xlsx')

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        features = extract_features(face_img)

        prediction = svm_model.predict([features])
        proba = svm_model.predict_proba([features]).max()
        proba = round(proba, 2)
        person = label_encoder.inverse_transform(prediction)[0]

        if proba > 0.3:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            text = f'{person} ({proba}%)'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if proba > 0.6:
                # Add the detection results to the DataFrame
                date = datetime.datetime.now().strftime("%m/%d/%Y")
                time = datetime.datetime.now().strftime("%I:%M:%S %p")

                # Check if the person already exists in the DataFrame
                existing_record = df[df['Person'] == person]

                if not existing_record.empty:
                    # If the person exists and the new probability is higher, update the record
                    if proba > existing_record['Probability'].values[0]:
                        df.loc[existing_record.index, 'Probability'] = proba
                else:
                    # If the person doesn't exist, append the new record
                    df = df._append({'Person': person, 'Date': date, 'Time': time, 'Probability': proba}, ignore_index=True)

    # Save the DataFrame to an Excel file
    if not df.empty:
        df.to_excel('attendance.xlsx', index=False)
    return frame

def extract_features(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (80, 80))
    image = (image / 255.0 - 0.5) * 2.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        features = facenet_model(image).cpu().numpy().flatten()
    return features

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def select_camera():
    num_cameras = 0
    camera_info = []
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            num_cameras += 1
            camera_name = cap.get(cv2.CAP_PROP_POS_MSEC)
            camera_info.append((i, camera_name))
            cap.release()
        else:
            break

    print("Cameras:")
    for camera_id, camera_name in camera_info:
        print(f"[{camera_id}] {camera_name}")

    while True:
        camera_id = input("Camera #: ")
        try:
            camera_id = int(camera_id)
            if 0 <= camera_id < num_cameras:
                return camera_id
            else:
                print("Invalid Camera!")
        except ValueError:
            print("Invalid Input!")

if __name__ == '__main__':
    app.run(debug=True)
