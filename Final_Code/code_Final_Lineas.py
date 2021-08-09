# To use the code it is necessary to install some libraries, for this the installation commands are:

# python3 -m pip install --upgrade pip
# OpenCV : pip3 install opencv-python
# Numpy : pip3 install numpy
# Mediapipe :  pip3 install mediapipe
# keyboard : pip3 install keyboard
# pytorch : pip3 install opencv-python torchvision

import cv2 
import numpy as np
import time
import os
import HandTrackingModule as htm
import keyboard 
import torch
import torchvision
from torchvision import datasets, transforms

test_dirs = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': '+',
    '11': '-',
    '12': '*',
    '13': '/',
    '14': '=',
    '15': 'a',
    '16': 'c'
}

#We load the trained model
model_nuevo = torch.load('./my_model_45.pt')
model_nuevo.eval()

brushThickness = 15
eraserThickness = 75
m=[]
#We load the necessary images for the interface
fondo = cv2.imread('Fondo.png')

folderPath = 'Header-Files'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 255)

#We declare the video capture
cap = cv2.VideoCapture(1)

cap.set(3,1280)
cap.set(4,720)

#We declare the hand detector by calling the "HandTrackingModule.py" functions.
detector = htm.handDetector(detectionCon=0.65,maxHands=1)
xp, yp = 0, 0

#we create the drawing canvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

flag = True


while True:

    # We obtain the frame in real time from the camera
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # We find the characteristic points of the hand together with its position
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #declare the coordinates of the index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # We use another function of the code to detect extended fingers
        fingers = detector.fingersUp()

        # Enter drawing selection mode: index finger extended
        if fingers[1] and fingers[2]==False:
            xp, yp = 0, 0
            # We declare the colors in selection mode and with ranges in the image of the drawing interface
            if y1 < 125:
                if 450 < x1 < 650:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 700 < x1 < 900:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 950 < x1 < 1050:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1100 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            # a rectangle is drawn that shows us the chosen color
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
    
        # Enter drawing mode: index and middle fingers extended
        if fingers[1]==True and fingers[2]==True and fingers[0]==False and fingers[3]==False and fingers[4]==False:
            # a circle is drawn to indicate that it is in drawing mode
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # coordinates for drawing are initialized 
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
        
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            # The thickness of the draft is declared and if not, the thickness of the drawing line is declared and the drawing line is made
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        
            xp, yp = x1, y1
    
        # Clean the entire canvas if only the little finger is extended 
        if  fingers[0]==False and fingers[1]==False and fingers[2]==False and fingers[3]==False and fingers[4]==True:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            
    # Image processing to obtain the binarized image, conversion to grayscale, binarized and superimposed of the image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    # The interface image of the drawing area is entered 
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img,0.8,imgCanvas,1,0)

    # The interface is resized to have a resolution of 1920x1080
    img = cv2.resize(img, (960,540), interpolation = cv2.INTER_AREA)
    imgInv1 = cv2.resize(imgInv, (960,540), interpolation = cv2.INTER_AREA)
    # the images are concatenated to obtain the final interface
    concat_horizontal = cv2.hconcat([img, imgInv1])
    if flag ==True:
        imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
        flag = False
    concat_Final = cv2.vconcat([concat_horizontal, imgText])

    #The conditional for the prediction starts
    if keyboard.is_pressed("f"): 
        #an auxiliary image of the binarized is created and processed
        img1 = imgInv
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        print((img1).shape)
        
        # We create a color mask only for the numbers
        lower_obj1 = np.array([0,0,0]) 
        upper_obj1 = np.array([1,1,1])
        
        obj1_mask1 = cv2.inRange(img1, lower_obj1, upper_obj1)
        
        # We count the number of numbers
        contours, _ = cv2.findContours(obj1_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dic_numbers = {}
        print('Numbers:', len(contours))
        
        for t in contours:
            x,y,w,h = cv2.boundingRect(t)
            img = cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        
        # The pixel x of all the images is counted to know the location of all
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            # The coordinate x and image of each contour are stored in a dictionary
            ROI = obj1_mask1[y:y+h, x:x+w]
            dic_numbers.update({x: ROI})
        
        # We print the x coordinate of all contours
        for key in dic_numbers:
          print(key)
        
          #  Sort dictionary
        from collections import OrderedDict
        dict1 = OrderedDict(sorted(dic_numbers.items()))
        
        # We store the ordered dictionaries and print them
        from PIL import Image
        img_total = []
        
        for key in dict1:
          img_total.append(dict1[key])
          print(key)
        
          # Save all cropped images
        x = range(0, len(img_total))
        for n in x:
          im = Image.fromarray(img_total[n])
          im.save("input_"+str(n)+"_number.png")
        
        from PIL import Image
        # Function to create a square image
        def make_square(im, min_size=10):
            x, y = im.size
            size = max(min_size, x, y)
            new_im = Image.new('RGB', (size, size))
            new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
            return new_im
        
        # Save all cropped images in square fortato
        x = range(0, len(img_total))
        for n in x:
          test_image = Image.open("input_"+str(n)+"_number.png")
          new_image = make_square(test_image)
          # Image save function
          new_image.save("sqr"+str(n)+"_img.png")
        
          from PIL import Image, ImageOps
        # Resize image to NN size 28x28
        size=(20,20)
        
        x = range(0, len(img_total))
        for n in x:
          im = Image.open("sqr"+str(n)+"_img.png")
          out = im.resize(size)
          out.save(str(n)+"_img.png") 
          # Add 7 borders to image
          ImageOps.expand(Image.open(str(n)+"_img.png"),border=4,fill='black').save(str(n)+"_img.png") 
        q=0
        x = range(0, len(img_total))
        s = ''
        total = ''
        for n in x:
            
            img1 = cv2.imread(str(n)+'_img.png')
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            # define the transform for the normalized
            elpepe = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                                ])
            elpepe_img = elpepe(gray1)
            print(elpepe_img.shape)

            img = elpepe_img.view(1, 784)
        
            with torch.no_grad():
                logps = model_nuevo(img)
        
            # The output of the network are logarithmic probabilities, it is necessary to take exponential for the probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            # the prediction is exchanged for the values ​​of the initial dictionary and these are concatenated
            for key in test_dirs:
                if (key) == str(probab.index(max(probab))): 
                    s = test_dirs[key]
            total = total + s
            n_s = total.replace("--", "=")
            
            print("Predicted Digit =", probab.index(max(probab)))
        m.append(n_s)
        print(n_s)
        print("list",len(m))
        #the string is entered as text in the output image and the image is concatenated again with the interface
        imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
        for p in range(len(m)):
            imgText =cv2.putText(imgText, m[p], (10, 180+q), cv2.FONT_HERSHEY_PLAIN, 3,(0, 0, 0), 3)
            #print(p)
            q=q+40
        concat_Final = cv2.vconcat([concat_horizontal, imgText])
        time.sleep(0.1)
    if keyboard.is_pressed("h"):
        q=0
        if (len(m))>1:
            m.pop(len(m)-1)
            imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
            for p in range(len(m)):
                imgText =cv2.putText(imgText, m[p], (10, 180+q), cv2.FONT_HERSHEY_PLAIN, 3,(0, 0, 0), 3)
                #print(p)
                q=q+40
            concat_Final = cv2.vconcat([concat_horizontal, imgText])
        else:
            pass
        time.sleep(0.1)
    # Pressing the "g" key clears the output image text
    if keyboard.is_pressed("g"):
        m=[]
        imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
        concat_Final = cv2.vconcat([concat_horizontal, imgText])
        time.sleep(0.01)
    #Finally the complete interface is shown as output
    #print(m)
    cv2.imshow('Canvas', concat_Final)
    cv2.waitKey(1)