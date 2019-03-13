import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load("recognizer/trainingData.yml")
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
path='DataSet'

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile



cam=cv2.VideoCapture(0);
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer/trainingData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):
    ret,img=cam.read();

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
            id,conf= recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            profile=getProfile(id)
            if(profile!=None):
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]),(x,y+h+60),font,255);
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[2]),(x,y+h+120),font,255);
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[3]),(x,y+h+180),font,255);
    cv2.imshow("face",img);
            
    if(cv2.waitKey(1)==ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()

   
