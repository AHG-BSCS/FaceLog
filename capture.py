import cv2
import os
import time

i = 0
cam = cv2.VideoCapture(0)
nameID = input("Name : ")
os.makedirs('faces/'+nameID)

while True:
    _, frame = cam.read()
    i+=1
    time.sleep(2)
    name ='faces/'+nameID+'/'+ str(i) + '.jpg'
    cv2.imwrite(name, frame)
    #frame = face_box(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord("q") or i > 50:
        break
cam.release()
cv2.destroyAllWindows()

'''
font = cv2.FONT_HERSHEY_SIMPLEX
    if len(face) != 0:
        cv2. putText(img, 'Scanning', (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
        #bbox = face[0]['box']
        #kp = face[0]['keypoints']
        #cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+ bbox[3]), (0,155,255),2)
    return img
'''


