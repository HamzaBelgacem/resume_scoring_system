import numpy as np
import cv2
#import pickle
from keras.models import load_model
#import urllib.request
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
model=load_model("model.h5")

font=cv2.FONT_HERSHEY_SIMPLEX
#url='http://192.168.43.1:8080/shot.jpg'
# IMPORT THE TRANNIED MODEL
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'Women'
    elif classNo == 1: return 'Men'

while True:
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32,32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)
    if probabilityValue > 0.8:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)),
         (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%",
         (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            
        if getCalssName(classIndex)=="Men":
            print("Not Organized CV")
        if getCalssName(classIndex)=="Women":
            print("Organized CV")      
    cv2.imshow("Result", imgOrignal)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllwindows()
