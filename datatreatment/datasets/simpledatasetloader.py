#Load small image datasets that can fit into memory
#Return Images (raw pixels intensities)and labels

import numpy as np
import cv2
import os

class SimpleDataSetLoader:
    def __init__(self, preprocessors=None):
        #store image preprocessor
        self.preprocessors = preprocessors

        #if emptu, initialize as empty list
        if self.preprocessors is None:
            self.preprocessors=[]
    
    def load(self, imagePaths, verbose=1):
        data=[] #list of features
        labels=[]

        for (i, imagePath) in enumerate(imagePaths):
            #load image and extract the class
            #path/{class}/image.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                #apply each preprocessors
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)
            
            size = len(imagePaths)
            if verbose>0 and i>0 and (i+1)%verbose == 0:
                print("processando {}/{}".format(i+1, size))
            
        return (np.array(data), np.array(labels))
