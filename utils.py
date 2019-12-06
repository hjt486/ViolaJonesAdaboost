#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2019

@author: Jiatai Han

This contains utility methods to support Viola-Jones face detection algorithm.

"""
import pickle
from ViolaJonesFaceDetector import *
import numpy as np
import os
import json
import cv2
import random

def getImages(folder):
    
    """ 
        Function to Retreive the names of Img files with Extension .png from the desired Folder
    """
    
    images = []
    
    for filename in os.listdir(folder):
        
        if filename.endswith('.png'):
            images.append(filename)
            
    images.sort()
    return images
        
        
def nonMaximalSupression(bounding_boxes, maxOverLap):
    
    """
        As we scale the features to match the Image and compare with different sizes, Multiple regions of a face may be detected by the classifier.
        
        Inorder to make sure the same face is not produced at several instances in the result, we have to suppress the result using Non Maximal Suppression Algorithm,as below.
        
        Reference: https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
        
    """
    if len(bounding_boxes) == 0: 
        return []
    
    pick = []
    
    x1 ,y1,x2, y2= bounding_boxes[:,0], bounding_boxes[:,1],bounding_boxes[:,2],bounding_boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    indexes = np.argsort(y2)
    
    while len(indexes) > 0:

        last = len(indexes) - 1
        
        i = indexes[last]
        
        pick.append(i)
        
        suppress = [last]
        
        for pos in range(0, last):
            
            j = indexes[pos]
 
            xx1 = max(x1[i], x1[j])
            
            yy1 = max(y1[i], y1[j])
            
            xx2 = min(x2[i], x2[j])
            
            yy2 = min(y2[i], y2[j])
            
            w = max(0, xx2 - xx1 + 1)
            
            h = max(0, yy2 - yy1 + 1)
 
            overlap = float(w * h) / area[j]
 
            if overlap > maxOverLap:
                
                suppress.append(pos)
 
        indexes = np.delete(indexes, suppress)
 
        return bounding_boxes[pick]
    

def TrainDetector(NoOfClassifiers, generate_features, faces_images_count, non_faces_images_count, window_size):
    """ 
        Method to Initialize the Training Process
    """
    classifier = FaceDetector(NoOfClassifiers)
    
    with open('Training-Data.pkl', 'rb') as f:
        
        training = pickle.load(f)
    
    classifier.Train(training, generate_features, faces_images_count, non_faces_images_count, window_size)
    
    classifier.save('Classifier')

def TestDetector(faces_images_count, non_faces_images_count):
    """ 
        Method to Initialize the Training Process
    """
    with open('Test-Data.pkl', 'rb') as f:
        Test = pickle.load(f)
    facedetector = FaceDetector()
    facedetector = facedetector.Load('Classifier')
    #print('Total:', len(facedetector.classifiers), ' Classifiers')
    i = len(facedetector.classifiers)
    fp = 0
    fn = 0
    accuracy = 0
    for image in Test:
        result = facedetector.classify(image[0])
        if result == image[1]:
            accuracy += 1
        if image[1] == 0 and result == 1:
            fp += 1
        if image[1] == 1 and result == 0:
                fn += 1

    print(facedetector.infos)
    print(facedetector.alphas)
    top_index = facedetector.alphas.index(max(facedetector.alphas))
    feature = facedetector.infos[top_index][0]
    top_threshold = facedetector.infos[top_index][1]
    training_accuracy, training_samples = facedetector.infos[top_index][2][0], facedetector.infos[top_index][2][1]
    facedetector.getBoxImage(facedetector.infos[top_index][0], i)
    print('---------------Test Result------------')
    print('Adaboost', i, 'Round(s)')
    print('Top Feature: ', feature[0], '(', feature[1] - 1, ',', feature[2] - 1,  ')', 'Width', feature[3], 'Height', feature[4])
    print('Top Threshold: {:.2}'.format(top_threshold))
    print('Top Training Accuracy: {:.2%}'.format(training_accuracy/training_samples), '(', training_accuracy, '/',training_samples,')')
    print('Test Accuracy: {:.2%}'.format(accuracy/len(Test)), '(', accuracy, '/', len(Test),')')
    print('Test False Positive: {:.2%}'.format(fp/len(Test),), '(', fp, '/', len(Test),')')
    print('Test False Negative: {:.2%}'.format(fn/len(Test),), '(', fn, '/', len(Test),')')
    print('---------------------------------------')

def TrainData(faces_path = './dataset/trainset/faces/', non_faces_path = './dataset/trainset/non-faces/', faces_size = 500, non_faces_size = 2500):
    
    """ Retreiving face and non-face images and storing them in a Pickle File for trainning
    
        Once this pickle file is generated we don't have to retreive images or convert them to gray scale again
        
        Until or unless you want to change the Training Data Set
    """
    
    Training_Data = []
    
    i = 1
    for filename in os.listdir(faces_path):
        
        if filename.endswith(".png"):
            if i > faces_size:
                break
            img=cv2.imread(faces_path+filename)
            
            # Comment out if resize is needed
            #img = cv2.resize(img,dsize=(19,19))
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            # Normalization
            img = np.array(img)
            mean = img.mean()
            std = img.std()
            img = (img - mean) / std 
            mean = img.mean()
            std = img.std()
            img = np.ndarray.tolist(img)
            
            data = (img,1)
            
            Training_Data.append(data)
            i += 1
    
    faces_images_count = len(Training_Data)

    i = 1        
    for filename in os.listdir(non_faces_path):
        
        if filename.endswith(".png"):
            if i > non_faces_size:
                break

            img=cv2.imread(non_faces_path+filename)
            
            #img = cv2.resize(img,dsize=(19,19))
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            # Normalization
            img = np.array(img)
            mean = img.mean()
            std = img.std()
            img = (img - mean) / std 
            mean = img.mean()
            std = img.std()
            img = np.ndarray.tolist(img)
            
            data = (img,0)
            
            Training_Data.append(data)
            
            i += 1
            
    random.shuffle(Training_Data)
    print(faces_images_count, 'faces images and', len(Training_Data)- faces_images_count, 'non-faces images are loaded and saved in', 'Training-Data.pkl')
        
    Training = open('./Training-Data.pkl','wb')
        
    pickle.dump(Training_Data,Training)

    return faces_images_count, len(Training_Data)- faces_images_count
        
    Training.close()
        
def TestData(faces_path = './dataset/testset/faces/', non_faces_path = './dataset/testset/non-faces/', faces_size = 500, non_faces_size = 2500):
    
    """ Retreiving face and non-face images and storing them in a Pickle File for testing
    
        Once this pickle file is generated we don't have to retreive images or convert them to gray scale again
        
        Until or unless you want to change the Training Data Set
    """

    Test_Data = []

    i = 1

    for filename in os.listdir(faces_path):
        
        if filename.endswith(".png"):
            if i > faces_size:
                break
            img=cv2.imread(faces_path+filename)
            
            # Comment out if resize is needed
            #img = cv2.resize(img,dsize=(19,19))
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            # Normalization
            img = np.array(img)
            mean = img.mean()
            std = img.std()
            img = (img - mean) / std 
            mean = img.mean()
            std = img.std()
            img = np.ndarray.tolist(img)
            
            data = (img,1)
            
            Test_Data.append(data)
            i += 1
    
    faces_images_count = len(Test_Data)

    i = 1        
    for filename in os.listdir(non_faces_path):
        
        if filename.endswith(".png"):
            if i > non_faces_size:
                break

            img=cv2.imread(non_faces_path+filename)
            
            # Comment out if resize is needed
            #img = cv2.resize(img,dsize=(19,19))
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            
            # Normalization
            img = np.array(img)
            mean = img.mean()
            std = img.std()
            img = (img - mean) / std 
            img = np.ndarray.tolist(img)

            data = (img,0)
            
            Test_Data.append(data)
            
            i += 1
            
    random.shuffle(Test_Data)
    print('Test dataset loaded')
    print(faces_images_count, 'faces images and', len(Test_Data)- faces_images_count, 'non-faces images are loaded and saved in', 'Test-Data.pkl')
        
    Test = open('./Test-Data.pkl','wb')
        
    pickle.dump(Test_Data,Test)

    return faces_images_count, len(Test_Data)- faces_images_count
        
    Test.close()
    
 
