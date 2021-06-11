import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import keyboard #Using module keyboard
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

model_nuevo = torch.load('./my_model_45.pt')
model_nuevo.eval()

brushThickness = 5
eraserThickness = 75

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

cap = cv2.VideoCapture(1)

cap.set(3,1280)
cap.set(4,720)
detector = htm.handDetector(detectionCon=0.65,maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

flag = True

while True:

    
    success, img = cap.read()
    img = cv2.flip(img, 1)

    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # print(lmList)
        
        
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        #print(fingers)

        
        if fingers[1] and fingers[2]==False:
            xp, yp = 0, 0
            #print('Selection Mode')
            
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
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
    
        
        if fingers[1]==True and fingers[2]==True and fingers[0]==False and fingers[3]==False and fingers[4]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print('Drawing Mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
        
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
        
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        
            xp, yp = x1, y1
    
        
        if  fingers[0]==False and fingers[1]==False and fingers[2]==False and fingers[3]==False and fingers[4]==True:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img,0.8,imgCanvas,1,0)
    img = cv2.resize(img, (960,540), interpolation = cv2.INTER_AREA)
    imgInv1 = cv2.resize(imgInv, (960,540), interpolation = cv2.INTER_AREA)
    concat_horizontal = cv2.hconcat([img, imgInv1])
    if flag ==True:
        imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
        flag = False
    concat_Final = cv2.vconcat([concat_horizontal, imgText])
    
    if keyboard.is_pressed("f"): 
        
        ######################################################################
        img1 = imgInv
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        print((img1).shape)
        
        # Creamos una mascara de color solo para los numeros
        lower_obj1 = np.array([0,0,0]) 
        upper_obj1 = np.array([1,1,1])
        
        obj1_mask1 = cv2.inRange(img1, lower_obj1, upper_obj1)
        
        # Contamos el numero de numeros
        contours, _ = cv2.findContours(obj1_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dic_numbers = {}
        print('Numbers:', len(contours))
        
        for t in contours:
            x,y,w,h = cv2.boundingRect(t)
            img = cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        
        
        # Se cuenta el pixel x de todas las imagenes para saber la ubicacion de todas
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            # Se almacena en un diccionario la cordenada x e imagen de cada contorno 
            ROI = obj1_mask1[y:y+h, x:x+w]
            dic_numbers.update({x: ROI})
        
        #Imprimimos la coordenada x de todos los contornos
        for key in dic_numbers:
          # print(dict1[key])
          print(key)
        
          # Ordenar diccionario 
        from collections import OrderedDict
        dict1 = OrderedDict(sorted(dic_numbers.items()))
        
        # Almacenamos los diccionarios ordenados y los imprimimos
        from PIL import Image
        img_total = []
        
        for key in dict1:
          # print(dict1[key])
          img_total.append(dict1[key])
          print(key)
        
          # Guardar todas las imagenes recortadas
        x = range(0, len(img_total))
        for n in x:
          im = Image.fromarray(img_total[n])
          im.save("input_"+str(n)+"_number.png")
        
        from PIL import Image
        # Funcion para crear una imagen cuadrada 
        def make_square(im, min_size=10):
            x, y = im.size
            size = max(min_size, x, y)
            new_im = Image.new('RGB', (size, size))
            new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
            return new_im
        
        # Guardar todas las imagenes recortadas en fortato cuadrado
        x = range(0, len(img_total))
        for n in x:
          test_image = Image.open("input_"+str(n)+"_number.png")
          new_image = make_square(test_image)
          # Funcion para guardar imagen
          new_image.save("sqr"+str(n)+"_img.png")
        
          from PIL import Image, ImageOps
        # Rejustar imagen a dimension de la CNN 28x28
        size=(20,20)
        
        x = range(0, len(img_total))
        for n in x:
          im = Image.open("sqr"+str(n)+"_img.png")
          out = im.resize(size)
          out.save(str(n)+"_img.png") 
          # Añadir bordes de 7 a la imagen
          ImageOps.expand(Image.open(str(n)+"_img.png"),border=4,fill='black').save(str(n)+"_img.png") 
        #################################################################
        m=[]
        x = range(0, len(img_total))
        o=0
        s = ''
        total = ''
        for n in x:
            
            img1 = cv2.imread(str(n)+'_img.png')
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            # Define a transform to normalize the data
            elpepe = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                                ])
            elpepe_img = elpepe(gray1)
            print(elpepe_img.shape)

            img = elpepe_img.view(1, 784)
        
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model_nuevo(img)
        
            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            for key in test_dirs:
                if (key) == str(probab.index(max(probab))): 
                    #print(key)
                    s = test_dirs[key]
                #print(n_s)
            total = total + s
            n_s = total.replace("--", "=")
            #print("Predicted Digit =", probab.index(max(probab)))
            #m=probab.index(max(probab))
            print("Predicted Digit =", probab.index(max(probab)))
            imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
            imgText =cv2.putText(imgText, n_s, (10, 350), cv2.FONT_HERSHEY_PLAIN, 10,(0, 0, 0), 10)
            concat_Final = cv2.vconcat([concat_horizontal, imgText])
            time.sleep(0.01)
            o=o+1
        
            
    if keyboard.is_pressed("g"): 
        imgText = cv2.resize(fondo, (1920,540), interpolation = cv2.INTER_AREA)
        concat_Final = cv2.vconcat([concat_horizontal, imgText])
        time.sleep(0.01)
    
    #cv2.imshow('Image', img)
    cv2.imshow('Canvas', concat_Final)
    #cv2.imshow('Inv', imgInv)
    cv2.waitKey(1)