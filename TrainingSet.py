#-----------------------------------------------------------------------------#
# Reference taken from MURTAZA HASSAN'S Code (Check out his youtube           #
# channel for informative Videos on Training CNN Models and Much More)        #
#-----------------------------------------------------------------------------#

#---------------------------#
# @author : Abhineet Sharma #
#---------------------------#

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

#-------------------  Parameters  ---------------------#
path='mydata'     # Where the Training dataset is stored
test_ratio = 0.2  # Splitting ratio : ( 80:20 )
validation_ratio = 0.2
imageDimensions = (32,32,3)
batchSizeVal = 50
epochsVal = 21    # The number of Epochs #
stepsPerEpochVal = 2000
#------------------------------------------------------#

images = []
classNo = []
myList=os.listdir(path)

print("Number Of Classes ",myList)
print(len(myList))

#-------------------------------------------------#
# Reading The Images of ASL Data_Set File by File #
# And appending them in an image list.            #
#-------------------------------------------------#
NoOfClasses=len(myList)
print("Importing Classes....")
for x in range(0,NoOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        currentImage = cv2.imread(path+"/"+str(x)+"/"+y)
        currentImage = cv2.resize(currentImage,(32,32))
        images.append(currentImage)
        classNo.append(x)
    print(x,end = " ")


print(len(images))
print(len(classNo))


images = np.array(images)
classNo = np.array(classNo)

#print(images.shape)
#print(classNo.shape)

#------------------------------------#
#    Splitting the freakin' Data     #
#------------------------------------#

X_train,X_test,Y_train,Y_test=train_test_split(images,classNo,test_size=test_ratio)
X_train,X_validation,Y_train,Y_validation=train_test_split(X_train,Y_train,test_size=validation_ratio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

num_of_samples=[]
for x in range(0,NoOfClasses):
    num_of_samples.append(len(np.where(Y_train == x)[0]))

print(num_of_samples)


#-------------------------------------------------------------#
# Prints a plot, denoting the number of images in each Folder #
#-------------------------------------------------------------#

plt.figure(figsize=(10,5))
plt.bar(range(0,NoOfClasses),num_of_samples)
plt.title("No of Images in Each Class")
plt.xlabel("Class ID")
plt.ylabel("No of Images")
plt.show()


#----------------------------------------------------------------#
# Pre - Processing the Image from data_set to a Gray Scale Image #
#----------------------------------------------------------------#
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#-------------------------------------------------------#
#               Reading only one image                  #
#-------------------------------------------------------#

#img = preProcessing(X_train[30])
#img = cv2.resize(img,(300,300))
#cv2.imshow("PreProcessed Image",img)
#cv2.waitKey(0)

#-------------------------------------------------------#
#       Reading only one image from Training data       #
#-------------------------------------------------------#

X_train=np.array(list(map(preProcessing,X_train)))
X_test=np.array(list(map(preProcessing,X_test)))
X_validation=np.array(list(map(preProcessing,X_validation)))
#img = X_train[30]
#img = cv2.resize(img,(300,300))
#cv2.imshow("PreProcessed Image",img)

#------------------------------------------------------------#
#     Changing the shape of the date by adding a channel     #
#------------------------------------------------------------#

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

#-------------------------------------------------------------#
#     Augmenting Images to make the DataSet more Generic      #
#-------------------------------------------------------------#
# width_shift_range  = 0.1 will give 10% width shift
# height_shift_range = 0.1 will give 10% height shift
# Shear_zoom will help tilt the images about X or Y axis
dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
dataGen.fit(X_train)

#----------------------------------------------#
#       Implementing One - Hot Encoding        #
#----------------------------------------------#

Y_train=to_categorical(Y_train,NoOfClasses)
Y_test=to_categorical(Y_test,NoOfClasses)
Y_validation=to_categorical(Y_validation,NoOfClasses)

#----------------------------------------------#
#             Designing the Model              #
#----------------------------------------------#

# Building of the Neural Network : Will place layer by layer
# Layer-By_layer building is made possible due to the Sequential API
# Number of perceptrons fot the model = 500
def myModel():
    #-----------------------------------------------------#
    # The Given Parameters are relative and Purely Depend #
    #            Upon the Model's Requirement             #
    #-----------------------------------------------------#

    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    #To build the model layer-by-layer (Sequential API)

    #layer 1
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    #layer 2
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))

    #Max pooling (layer 3)
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    #Layer 4 using second filter
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))

    #Layer 5 using second filter
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NoOfClasses, activation='softmax'))
    #--------------------------------------------------------#
    # Adam is Adaptive Gradient Algorithm which helps in     #
    # Adaptive Moment Optimization and handles the problem   #
    # of Vanishing Gradient and Exploding Gradient           #
    #--------------------------------------------------------#

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()

print(model.summary())

history = model.fit_generator(dataGen.flow(X_train,Y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,Y_validation),
                                 shuffle=1)
#----------------------------------------------#
# Plots for visualizing the Accuracy and Error #
#          of the model every EPOCH            #
#----------------------------------------------#

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.show()

#-----------------------------------------------------#
#   Displaying the Model's Test Score and Accuracy    #
#-----------------------------------------------------#

score=model.evaluate(X_test,Y_test,verbose=0)
print('Test Score = ',score[0])
print('Accuracy = ',score[1])


#----------------------------------------------------#
#     Creating a Pickle Object to save the Model     #
#----------------------------------------------------#

pickle_out=open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()


cv2.waitKey(0)
#------------------------------------------------#
#                 End of Script                  #
#------------------------------------------------#