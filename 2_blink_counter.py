import numpy as np 
import cv2

#The goal here is just to pick a image and put a blue square over the face. Pretty simple. 
#O Objetivo aqui é só encontrar alguns rostos em algumas mensagens.

#CascadeClassifiers
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

first_read = True

#Video Capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()

while(ret):
    ret,img = cap.read()

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1 ,1)

    faces = face_cascade.detectMultiScale(gray, 1.3,5,minSize=(200,200))
    print(faces)
    
    if(len(faces)>0):
        for(x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) 

            #roi_face
            roi_face = gray[y:y+h, x:x+w]
            roi_face_clr = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,  1.3,5,minSize=(50,50))

            #Examining 
            if(len(eyes)>2):
                if(first_read):
                    cv2.putText(img, "Eyes detected. Press X to begin", (70,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
                else:
                    cv2.putText(img, "Eyes open", (70,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
            else:
                if(first_read):
                    cv2.putText(img, "No eyes detected.", (70,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
                else:
                    cv2.putText(img, "Eyes open", (70,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

    else:
         cv2.putText(img, "No face detected.", (170,170), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)

    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord('x') and first_read):
        first_read = False

cap.release()
cv2.destroyAllWindows()