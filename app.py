import os
import cv2
import pickle
from flask import Flask, render_template, Response, request, jsonify
from threading import Thread
import torch
from facenet_pytorch import InceptionResnetV1
import time

# Load pre-trained SVM model and label encoder
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

app = Flask(__name__)
camera = None
cameramode = None

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/face_recognition')
def face_recognition():
    global camera
    global cameramode

    if camera is None:
        camera = VideoCamera()
        cameramode = 'recognition'
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_capturing')
def face_capturing():
    global camera
    global cameramode
    
    if camera is None:
        camera = VideoCamera()
        cameramode = 'capture'
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global camera
    global cameramode

    if camera is not None:
        camera.__del__()
        camera = None
        cameramode = None
    return 'Webcam stopped'

@app.route('/capture_images', methods=['POST'])
def capture_images():
    global camera
    global cameramode

    user_name = request.json.get('user_name')
    if not user_name:
        return jsonify({"error": "User name is required"}), 400
    
    # Create a folder for the user if it doesn't exist
    user_folder = os.path.join('faces', user_name)
    os.makedirs(user_folder, exist_ok=True)

    # Capture 30 images at 0.8-second intervals
    count = 0
    while count < 50:
        if camera is None:
            break
        frame = camera.get_frame_as_image()
        file_path = os.path.join(user_folder, f'{count + 1}.jpg')
        cv2.imwrite(file_path, frame)
        
        camera.set_flash()
        count += 1
        cv2.waitKey(800)

    return jsonify({"message": "Images captured"}), 200

# Threading for video capture and processing
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.grabbed, self.frame = self.video.read()
        self.last_recognition_time = time.time()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.flash = False

    def update(self):
        while True:
            self.grabbed, self.frame = self.video.read()
            if self.flash:
                self.frame[:] = 0  # Set frame to black
                self.flash = False

    def get_frame(self):
        frame = self.frame.copy()
        current_time = time.time()

        # Perform face recognition every 0.1 seconds
        if current_time - self.last_recognition_time >= 0.1:
            if cameramode == 'recognition':
                frame = recognize_faces(frame)
                self.last_recognition_time = current_time
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_frame_as_image(self):
        return self.frame.copy()

    def capture_frame(self):
        return self.frame.copy()
    
    def set_flash(self):
        self.flash = True
    
    def __del__(self):
        self.video.release()

def recognize_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        features = extract_features(face_img)

        prediction = svm_model.predict([features])
        proba = svm_model.predict_proba([features]).max()
        person = label_encoder.inverse_transform(prediction)[0]

        if proba > 0.7:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            text = f'{person} ({proba:.2f})'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

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

if __name__ == '__main__':
    app.run(debug=True)
