import cv2
import os
import time

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

    print("Available cameras:")
    for camera_id, camera_name in camera_info:
        print(f"{camera_id}: {camera_name}")

    while True:
        camera_id = input("Enter the number of the camera you want to use: ")
        try:
            camera_id = int(camera_id)
            if 0 <= camera_id < num_cameras:
                return camera_id
            else:
                print("Invalid camera number. Please enter a number between 0 and", num_cameras - 1)
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    camera_id = select_camera()
    cam = cv2.VideoCapture(camera_id)

    nameID = input("Name : ")
    os.makedirs('faces/' + nameID)

    i = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        i += 1
        time.sleep(2)
        name = 'faces/' + nameID + '/' + str(i) + '.jpg'
        cv2.imwrite(name, frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord("q") or i > 50:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
