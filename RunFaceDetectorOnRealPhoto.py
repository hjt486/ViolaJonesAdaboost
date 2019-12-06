#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2019

@author: Jiatai Han
"""
import cv2
import numpy as np
from ViolaJonesFaceDetector import FaceDetector
from utils import nonMaximalSupression,getImages

# Change your photo here
testimage = './read-world_photos/testimage8.jpg'

def getFaceLocations(gray):

    print('Start recognition...')
    
    rowssize = 50
            
    colssize = 50
                                    
    clf = FaceDetector.Load('Classifier')
            
    locations = []
            
    while rowssize<(len(gray)-2):
                
        for r in range(0,gray.shape[0] - rowssize, 10):
                    
            for c in range(0,gray.shape[1] - colssize, 10):
                        
                window = gray[r:r+rowssize,c:c+colssize]
                        
                window=cv2.resize(window,dsize=(19,19))
                        
                prediction = clf.classify(window)
                        
                if prediction ==1:
                                        
                    locations.append([r,c,r+rowssize,c+colssize])
                    
        colssize+=50
        
        rowssize+=50
    
    return locations
    
           
        
imgtest = cv2.imread(testimage)                    
gray = cv2.cvtColor(imgtest,cv2.COLOR_RGB2GRAY)
print('Image', testimage, ' is loaded')

# Normalize
mean = gray.mean()
std = gray.std()
std = gray.std()

#locations = [[1,2,3,4],[1,2,3,4]]
locations = getFaceLocations(gray)
            
if locations:
    locations = nonMaximalSupression(np.array(locations), 0)
    print('Face(s) recognized at', locations)
    for location in locations:
        cv2.rectangle(imgtest, (location[0], location[1]), (location[2], location[3]), (255,0,0), 2)
    cv2.imwrite("result.png",imgtest)
    cv2.imshow("result.png", imgtest)
    k = cv2.waitKey(0) # 0==wait forever
else:
    print('Face not recognized, please change to another photo.')


    
    
    
    
        
    
    
    

    
