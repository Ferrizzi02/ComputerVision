#load image and resize to a fixed size ignoring the aspect ratio

#For KNN: all images should have fixed feature vector size (identical size)

import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
       #store target image size and interpolation
       self.width = width
       self.height = height
       self.inter = inter #control the interpolation algorithm.
    

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)