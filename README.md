# Computer Vision Project:
Virtual Whiteboard

Adrian Flores Camacho
Andrés Saúl Bellido Gonzales
Alejandro Núñez Arroyo
Raul Angel Mollocuaquira Caparicona

# Abstract

---

The present work consists of the implementation of a hand recognition algorithm for tracing letters and numbers in a space visible to a computer camera in order to obtain a ”virtual whiteboard” and achieve a better explanation by a teacher or speaker in a video conference who needs to use a whiteboard and point with gestures while explaining the content of the ideas to be shared.

## Dataset

We looked for a different dataset to MNIST which was thebasis for presenting the results in Project milestone [5], welooked for the new dataset to allow us not only to have theinformation of numbers between 0 and 9, but also to containdifferent numeric characters and letters. For this we will usethe implementation of the Handwritten math symbols dataset,which has different classes of characters.

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled.png)

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%201.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%201.png)

## Hand detection and segmentation

Mediapipe was very useful to us to add value to our project,mediapipe has functions for face recognition, face meshing,object detection, face and hair segmentation, etc. The functionthat interested us for the project was the detection of hands,which allowed us to detect a hand next to the characteristicpoints, a function included in mediapipe is to obtain thecoordinates of these points that helped us to verify if a fingeris raised and if it is not, and this was very useful to performdrawing functions in the interface.

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%202.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%202.png)

## Virtual Whiteboard interface

There is the selection mode when only the index finger is extended, in this mode, you can select between 4 colors, with RGB values of (255,0,0), (0,255,0), (0,0,255), and (0,0,0), in the drawing mode that is entered with the index and middle fingers extended, with the colors and the ”cv2.line ()”function you can draw lines on the created canvas and on the same image obtained by the frames in the coordinates of the tip of the index finger. The last drawing function you have is to clean all the drawings that can be activated if only the pinkie finger is extended.

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%203.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%203.png)

## Character recognition

Theseimages are read and recognized by a trained model, whichdefines the class to which the trace belongs, outputting thepredicted character.Finally, this value predicted by the model is taken as theindex of the dictionary previously generated with the classes,and finally, all these values are concatenated to be able tobe put on another canvas but as text, the result is the following:

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%204.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%204.png)

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%205.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%205.png)

![Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%206.png](Computer%20Vision%20Project%20Virtual%20Whiteboard%207524d50bf82b4eedb654e1ef595ded60/Untitled%206.png)
