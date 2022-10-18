#importing the requirred libs
import cv2
import dlib
import numpy as np

######image preparation#########
#loading the model image
img=cv2.imread('Hermoine1.jpg')


#make copy from the original image
finalimage=img.copy()
#convert the image to grey
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#display the grey image
#greyimage: complexity is lesser, can obtain features relating to brightness,etc.Processing is also faster
cv2.imshow("gray image ", gray_img)
#view the image
cv2.imshow("original image",img)

#dlib variables preparation for face detection
#declare the detector from the dlib
detector=dlib.get_frontal_face_detector()
#load the predictor
#its a pretrained facial landmark dtetector,tool
path='shape_predictor_68_face_landmarks.dat'
predictor=dlib.shape_predictor(path)
#start to detect the face in the image
faces=detector(gray_img)
print(faces)

#(x1,y1)################################
#                                      #
#                                      #
#                                      #
#                                      #
#                                      #
#################################(x2,y2)
landmarkspoints=[]

#an empty function for the trackers
def empty(a):
    pass
#Adding track Bars for color selection
#Make window for the trackers
cv2.namedWindow("Color_Selection")
cv2.resizeWindow("Color_Selection",640,180)
#Red
cv2.createTrackbar("Red", "Color_Selection", 0, 255, empty)
#Green
cv2.createTrackbar("Green", "Color_Selection", 240, 255, empty)
#Blue
cv2.createTrackbar("Blue", "Color_Selection", 222, 255, empty)


for face in faces:
    x1,y1=face.left(),face.top()
    x2,y2=face.right(),face.bottom()
    img  = cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,0),3)
    cv2.imshow("face detected",img)
    #predict the face landmark
    landmarks=predictor(gray_img,face)
    for n in range(68):
        x=landmarks.part(n).x
        y=landmarks.part(n).y
        landmarkspoints.append([x,y])
        cv2.circle(img,(x,y),3,(0,255,0),cv2.FILLED)
        cv2.putText(img,str(n),(x+1,y-10),cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,255),1)
        #lips region is at 48-60
        #crop the lip region
    landmarkspoints=np.array(landmarkspoints)
    #return an array of zeroes with the same shape and type as given array
    lipmask=np.zeros_like(img)
    lipimg=cv2.fillPoly(lipmask, [landmarkspoints[48:60]],(255,255,255))
    cv2.imshow("lip",lipimg)
    cv2.imshow("Face landmark",img)
    lipimgcolor=np.zeros_like(lipimg)

    #set the manual color for the lip
    b=75
    g=55
    r=177
    lipimgcolor[:]=b,g,r
    cv2.imshow("lipimgcolor",lipimgcolor)
    lipimgcolor=cv2.bitwise_and(lipimg,lipimgcolor)
    #new colored lip
    lipimgcolor=cv2.GaussianBlur(lipimgcolor,(7,7),10)
    #add it to the original image
    finalmakeup=cv2.addWeighted(finalimage,1,lipimgcolor,0.6,0)
    cv2.imshow("final makeup", finalmakeup)

cv2.waitKey(0)