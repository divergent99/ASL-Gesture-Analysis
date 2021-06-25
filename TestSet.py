import numpy as np
import cv2
import pickle # To dump the saved model in the current Python Script

#-------------------- PARAMETERS -------------------#
width = 640
height = 480
threshold = 0.60 # MINIMUM PROBABILITY TO CLASSIFY
cNo = 0
#---------------------------------------------------#


#-------------------------------#
#    CREATING CAMERA OBJECT     #
#-------------------------------#
cap = cv2.VideoCapture(cNo)
cap.set(3,width)
cap.set(4,height)

#------------------------------#
#    LOAD THE TRAINED MODEL    #
#------------------------------#
pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)

#------------------------------#
#  PRE - PROCESSING FUNCTION   #
#---------------------------- -#
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #----------------------------#
    #   PREDICTING CLASS LABEL   #
    #----------------------------#
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if(classIndex==0):
        out='A'
    elif(classIndex==1):
        out='B'
    elif(classIndex==2):
        out='C'
    elif(classIndex == 3):
        out = 'D'
    elif (classIndex == 4):
        out = 'del'
    elif (classIndex == 5):
        out = 'E'
    elif (classIndex == 6):
        out = 'F'
    elif (classIndex == 7):
        out = 'G'
    elif (classIndex == 8):
        out = 'H'
    elif (classIndex == 9):
        out = 'I'
    elif (classIndex == 10):
        out = 'J'
    elif (classIndex == 11):
        out = 'K'
    elif (classIndex == 12):
        out = 'L'
    elif (classIndex == 13):
        out = 'M'
    elif (classIndex == 14):
        out = 'N'
    elif (classIndex == 15):
        out = ' Nothing'
    elif (classIndex == 16):
        out = 'O'
    elif (classIndex == 17):
        out = 'P'
    elif (classIndex == 18):
        out = 'Q'
    elif (classIndex == 19):
        out = 'R'
    elif (classIndex == 20):
        out = 'S'
    elif (classIndex == 21):
        out = 'space'
    elif (classIndex == 22):
        out = 'T'
    elif (classIndex == 23):
        out = 'U'
    elif (classIndex == 24):
        out = 'V'
    elif (classIndex == 25):
        out = 'W'
    elif (classIndex == 26):
        out = 'X'
    elif (classIndex == 27):
        out = 'Y'
    elif (classIndex == 28):
        out = 'Z'
    else:
        out = 'Unidentifiable'

    if probVal> threshold:
        cv2.putText(imgOriginal,out + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
#-------------------------------#
#         End of Script         #
#-------------------------------#