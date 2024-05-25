import os
import cv2

def face_filter():
    os.makedirs('facetest', exist_ok=True)
    faces_folder_dir = 'faces'
    folder_count = 0
    image_count = 0
    face_count = 0
    no_face = 0
    multiple_face = 0

    if not os.path.exists(faces_folder_dir):
        print("No faces found")
        return

    for person_name in os.listdir(faces_folder_dir):
        person_dir = os.path.join(faces_folder_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            
            if len(faces) == 0:
                print(f"No faces: {image_path}")
                no_face += 1
                os.remove(image_path)
                continue
            elif len(faces) > 1:
                print(f"Multiple faces: {image_path}")
                os.remove(image_path)
                multiple_face += 1
                continue

            # Save the faces from the image
            os.makedirs(f'facetest/{person_name}', exist_ok=True)
            for (x, z, w, h) in faces:
                face_img = image[z:z+h, x:x+w]
                cv2.imwrite(f'facetest/{person_name}/{image_name}', face_img)
                face_count += 1
            image_count += 1
        folder_count += 1

    print("")
    print(f"Folder searched: {folder_count}")
    print(f"Image processed: {image_count}")
    print(f"Faces saved: {face_count}")
    print(f"No faces found: {no_face}")
    print(f"Multiple face found: {multiple_face}")

face_filter()