import time 
import cv2

#The goal here is just to pick a image and put a blue square over the face. Pretty simple. 
#O Objetivo aqui é só encontrar alguns rostos em algumas imagens.

#CascadeClassifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#The face is generated from thispersondoesnotexist.com
#Os rostos deste mini-projeto vem do thispersondoesnotexist.com
img = cv2.imread('images/thispersondoesnotexist.jpg')

#convert into grayscale because computers don't work like humans, yet. They need to process stuff
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#Draw the square
#Desenhe o quadro
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) 

cv2.imshow('img', img)
cv2.waitKey()