#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2019

@author: Jiatai Han

This contains all classes and functions needed for Viola-Jones algorithm

"""
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
import os
import cv2
import matplotlib.pyplot as plt


def getIntegralImage(img):
    """
        Function to Compute the integral image of a given Image using the formula: 
            
        Integral_Img[i][j] = img[i][j] + integral[i-1][j] + integral[i][j-1] + - integral[i-1][j-1]

        Note that additional one row and one column of 0s are padded so no worry about edges
    """
    row = len(img)
    
    col = len(img[0])
    
    integral = np.zeros((row + 1,col +1))
    
    for i in range(1,row + 1):
        
        for j in range(1,col +1):
            
            integral[i][j] = int(img[i-1][j-1])
            
            if i-1 >=0 and j-1 >=0:
                
                integral[i][j] = integral[i][j] + integral[i-1][j] + integral[i][j-1] + - integral[i-1][j-1] 
                
            elif i-1 >= 0:
                
                integral[i][j] = integral[i][j] + integral[i-1][j]
                
            elif j-1 >= 0:
                
                integral[i][j] = integral[i][j] + integral[i][j-1]
            
    return integral

def getImage(filepath):
    
    """ 
        Function to Retreive the names of Img files with Extension .png from the desired Folder
    """
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return image

class Box:
    def __init__(self, type, x, y, width, height):

        """ 
        Generate box object to calcualte Haar's features with given coordiates and width and height

        """

        self.type = type
        
        self.x = x
        
        self.y = y
        
        self.width = width
        
        self.height = height
    
    def compute_feature(self, integralImg):
        """
            Computes the value of the Haar's feature given the integral image
        """
        result = None
        if self.type == 'twoVerticle':
            '''
            AB
            CD
            EF
            '''
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width - 1)
            C = (self.y + self.height//2 - 1, self.x - 1)
            D = (self.y + self.height//2 - 1, self.x + self.width - 1)
            E = (self.y + self.height - 1, self.x - 1)
            F = (self.y + self.height - 1, self.x + self.width - 1)
            result = 2 * integralImg[D[0]][D[1]] + integralImg[A[0]][A[1]] - integralImg[B[0]][B[1]] - 2 * integralImg[C[0]][C[1]] + integralImg[E[0]][E[1]] - integralImg[F[0]][F[1]]

        elif self.type == 'twoHorizontal':
            '''
            ABC
            DEF
            '''
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width//2 - 1)
            C = (self.y - 1 , self.x + self.width - 1)
            D = (self.y  + self.height - 1, self.x - 1)
            E = (self.y  + self.height - 1 , self.x + self.width//2 - 1)
            F = (self.y  + self.height - 1 , self.x + self.width - 1)
            result = 2 * integralImg[B[0]][B[1]] + integralImg[F[0]][F[1]] - integralImg[C[0]][C[1]] - 2 * integralImg[E[0]][E[1]] + integralImg[D[0]][D[1]] - integralImg[D[0]][D[1]]

        elif self.type == 'threeHorizontal':
            '''
            ABCD
            EFGH
            '''
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width//3 - 1)
            C = (self.y - 1 , self.x + self.width//3 * 2 - 1)
            D = (self.y - 1 , self.x + self.width - 1)
            E = (self.y  + self.height - 1 , self.x - 1)
            F = (self.y  + self.height - 1 , self.x + self.width//3 - 1)
            G = (self.y  + self.height - 1, self.x + self.width//3 * 2 - 1)
            H = (self.y  + self.height - 1, self.x + self.width - 1)
            result = 2 * integralImg[B[0]][B[1]] + 2 * integralImg[G[0]][G[1]] - 2 * integralImg[C[0]][C[1]] - 2 * integralImg[F[0]][F[1]] - integralImg[H[0]][H[1]] - integralImg[A[0]][A[1]] + integralImg[D[0]][D[1]] + integralImg[E[0]][E[1]]
        
        elif self.type == 'threeVerticle':
            '''
            AB
            CD
            EF
            GH
            '''
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width - 1)
            C = (self.y + self.height//3 - 1 , self.x - 1)
            D = (self.y + self.height//3 - 1 ,  self.x + self.width - 1)
            E = (self.y  +  self.height//3 * 2 - 1 , self.x - 1)
            F = (self.y  + self.height//3 * 2 - 1 , self.x + self.width - 1)
            G = (self.y  + self.height - 1, self.x - 1)
            H = (self.y  + self.height - 1, self.x + self.width - 1)
            result = 2 * integralImg[C[0]][C[1]] + 2 * integralImg[F[0]][F[1]] - 2 * integralImg[D[0]][D[1]] - 2 * integralImg[E[0]][E[1]] - integralImg[H[0]][H[1]] - integralImg[A[0]][A[1]] + integralImg[B[0]][B[1]] + integralImg[G[0]][G[1]]

        elif self.type == 'four':
            '''
            ABC
            DEF
            GHI
            '''
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width//2 - 1)
            C = (self.y - 1 , self.x + self.width - 1)
            D = (self.y + self.height//2 - 1, self.x - 1)
            E = (self.y + self.height//2 - 1, self.x + self.width//2 - 1)
            F = (self.y + self.height//2 - 1, self.x + self.width - 1)
            G = (self.y + self.height - 1, self.x - 1)
            H = (self.y + self.height - 1, self.x + self.width//2 - 1)
            I = (self.y + self.height - 1, self.x + self.width - 1)
            result = -integralImg[A[0]][A[1]] + 2 * integralImg[B[0]][B[1]] - integralImg[C[0]][C[1]] + 2 * integralImg[D[0]][D[1]] - 4 * integralImg[E[0]][E[1]] + 2 * integralImg[F[0]][F[1]] - integralImg[G[0]][G[1]] + 2 * integralImg[H[0]][H[1]] - integralImg[I[0]][I[1]]
        return((self.type, self.x, self.y, self.width, self.height), result)


class Classifier:
    def __init__(self, feature, threshold, polarity):
        """
            Initializes a Classifier
        """
        self.feature = feature
        
        self.threshold = threshold
        
        self.polarity = polarity
    
    def classify(self, x):
        """
            Classifies an integral image based on a feature f and the classifiers threshold and polarity 
        """
        feature = Box(self.feature[0], self.feature[1], self.feature[2], self.feature[3], self.feature[4])
        feature_value = feature.compute_feature(x)[1]
        if self.polarity * feature_value < self.polarity * self.threshold:
            return (1, self.feature, feature_value, self.threshold)
        else:
            return (0, self.feature, feature_value, self.threshold)
    

class FaceDetector:
    def __init__(self, NoOfAdaboost = 20):
        """
            Initializes the Face Detector with NoOfAdaboost (round of Adaboost)
        
        """
        self.NoOfAdaboost = NoOfAdaboost
        
        self.alphas = []

        self.infos = []
        
        self.classifiers = []

    def getBoxImage(self, best_feature, round):
        feature = best_feature[0]
        x = best_feature[1] - 1
        y = best_feature[2] - 1
        w = best_feature[3]
        h = best_feature[4]
        image = getImage('example.png')
        for i in range(len(image)):
            for j in range(len(image[0])):
                if feature == 'twoVerticle':
                    if i >= y and i < y + h//2 and j >= x and j < x + w:
                        image[i][j] = 255
                    elif i >= y + h//2 and i < y + h and j >= x and j < x + w:
                        image[i][j] = 0
                elif feature == 'twoHorizontal':
                    if i >= y and i < y + h and j >= x and j < x + w//2:
                        image[i][j] = 0
                    elif i >= y and i < y + h and j >= x + w//2 and j < x + w:
                        image[i][j] = 255
                elif feature == 'threeHorizontal':
                    if i >= y and i < y + h and j >= x and j < x + w//3:
                        image[i][j] = 0
                    elif i >= y and i < y + h and j >= x + w//3 and j < x + w//3 * 2:
                        image[i][j] = 255
                    elif i >= y and i < y + h and j >= w//3 * 2 and j < x + w:
                        image[i][j] = 0
                elif feature == 'threeVerticle':
                    if i >= y and i < y + h//3 and j >= x and j < x + w:
                        image[i][j] = 0
                    elif i >= y + h//3 and y + h//3 * 2  and j >= x and j < x + w:
                        image[i][j] = 255
                    elif i >= y + h//3 * 2 and i < y + h and j >= x and j < x + w:
                        image[i][j] = 0
                elif feature == 'four':
                    if i >= y and i < y + h//2 and j >= x and j < x + w//2:
                        image[i][j] = 0
                    elif i >= y + h//2 and i < y + h and j >= x and j < x + w//2:
                        image[i][j] = 255
                    elif i >= y and i < y + h//2 and j >= x + w//2 and j < x + w:
                        image[i][j] = 255
                    elif i >= y + h//2 and i < y + h and j >= x + w//2 and j < x + w:
                        image[i][j] = 0
        print('getBoxImage', best_feature)
        plt.figure()
        plt.imshow(image)
        str1 = 'Top Feature of ' + str(round) + ' Round(s) Adaboost'
        plt.suptitle(str1)
        str2 = 'Feature: ' + feature + ' at ( ' + str(x) + ',' + str(y) + ' ) width: ' + str(w) + ' height: ' +  str(h)
        plt.title(str2)
        plt.show()  # display it
        
    def Train(self, Training, generate_features, NoOfFaces, NoOfNonFaces, window_size):
        """
            Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
            
            Training: A List of tuples (Image, Classification (1 - if positive image 0 - if negative image) 
            
            NoOfFaces: the number of positive samples
            
            NoOfNonFaces: the number of negative samples
            
        """
        print('Training Started.......')
        
        weights = np.zeros(len(Training))
        
        training_data = []

        for x in range(len(Training)):
            
            training_data.append((getIntegralImage(Training[x][0]), Training[x][1]))

            if Training[x][1] == 1:
                
                weights[x] = 1.0 / (2 * NoOfFaces)
                
            else:
                
                weights[x] = 1.0 / (2 * NoOfNonFaces)


        if generate_features != 'new' and os.path.exists('features.pkl'):

            with open("features.pkl", 'rb') as f:

                print('Features existed and loaded')

                features, X, y, z = pickle.load(f)
        else:
            features = self.BuildFeatures(training_data[0][0].shape, window_size)

            X, y, z = self.apply_features(features, training_data)

            save = (features, X, y, z)

            with open("features.pkl", 'wb') as f:

                pickle.dump(save, f)
        
        #Speed up features selection using SelectPercentile as pre-selections
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        

        for t in range(self.NoOfAdaboost):
            
            weights = weights / np.linalg.norm(weights)
            
            weak_classifiers = self.TrainWeak(X, y, z, features, weights)

            best_threshold, best_feature, clf, error, accuracy, best_fp, best_fn = self.SelectBest(weak_classifiers, weights, training_data)

            beta = error / (1.0 - error)
   
            for i in range(len(accuracy)):
                
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
                
            alpha = math.log(1.0/beta)
            
            self.alphas.append(alpha)
            
            self.classifiers.append(clf)
            accuracy = (len(accuracy) - sum(accuracy))
            self.infos.append((best_feature, best_threshold, (accuracy, len(training_data))))
            print(accuracy)
            print('-------------Round #', str(t + 1), '-----------------')
            print('Alpha: {:.2}'.format(alpha))
            print('Feature: ', best_feature[0], '(', best_feature[1] - 1, ',', best_feature[2] - 1,  ')', 'Width', best_feature[3], 'Height', best_feature[4])
            print('Training Error: {:.2%}'.format(error))
            print('Training Treshold: {:.2}'.format(best_threshold))
            print('Training Accuray: {:.2%}'.format(accuracy/len(training_data)), '(', accuracy, '/', len(training_data),')')
            print('Training False Positive: {:.2%}'.format(best_fp/len(training_data),), '(', best_fp, '/', len(training_data),')')
            print('Training False Negative: {:.2%}'.format(best_fn/len(training_data),), '(', best_fn, '/', len(training_data),')')
            print('---------------------------------------')
        
        print('success: Training Done')
        print('---------------------------------------')

    def TrainWeak(self, X, y, z, features, weights):
        """
            Finding the optimal thresholds for each classifier given the current weights
        """
        print('Start training weak classifiers')
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            
            if label == 1:
                
                total_pos += w
                
            else:
                
                total_neg += w

        classifiers = []

        #print(X)

        for index, feature in enumerate(X):
                            
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            
            pos_weights, neg_weights = 0, 0
            
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            
            for w, f, label in applied_feature:
                
                #print('Threshold', label)
                
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                
                if error < min_error:
                    
                    min_error = error
                    
                    best_feature = z[index]
                    
                    best_threshold = f
                    
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    
                    pos_seen += 1
                    
                    pos_weights += w
                    
                else:
                    neg_seen += 1
                    
                    neg_weights += w
            
            clf = Classifier(best_feature, best_threshold, best_polarity)
            
            classifiers.append(clf)

        return classifiers
                
    def BuildFeatures(self, image_shape, window_size):
        """
            Builds the possible features given an image shape         
        """
        imgHeight, imgWidth = image_shape

        features = []
        
        print('Building features....')

        features=[]

        nums=[]
        
        featureTypes=[("twoVerticle",(1,2)),("twoHorizontal",(2,1)),
                    ("threeHorizontal",(3,1)),("threeVerticle",(1,3)),("four",(2,2))]

        for type,size in featureTypes:

            num=0

            for width in range(size[0],window_size+1,size[0]):

                for height in range(size[1],window_size+1,size[1]):

                    for x in range(1, imgWidth-width+1):

                        for y in range(1, imgHeight-height+1):

                            features.append(Box(type,x,y,width,height))

                            num+=1

            nums.append(num)
        print('Total features for', imgHeight - 1, 'x' ,imgWidth - 1, 'image with window size of', window_size, 'is', sum(nums))
        
        return np.array(features)

    def SelectBest(self, classifiers, weights, training_data):
        """
            Selects the best  classifier for the given weights
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        
        print('selecting best classifier out of '+ str(len(classifiers)))

        i = 1

        for classifier in classifiers:

            if i % (len(classifiers)//4) == 0:

                print('{:.2%}'.format(i/len(classifiers)), 'finished, please wait...')

            error, accuracy = 0, []

            fp = 0

            fn = 0

            for data, w in zip(training_data, weights):

                classify, feature, feature_value, threshold = classifier.classify(data[0])

                correctness = abs(classify - data[1])

                if data[1] == 0 and classify == 1:

                    fp += 1

                if data[1] == 1 and classify == 0:

                    fn += 1

                accuracy.append(correctness)
                
                error += w * correctness

            # Traditional error judging method given by the paper
            error = error / len(training_data)
            
            # New error judging method by false positives or false nagatives
            '''
            When lbd = 0 FP prefered
            Training Error: 100.00%
            Training Accuray: 84.83% ( 2120 / 2499 )
            Training False Positive: 0.04% ( 1 / 2499 )
            Training False Negative: 15.13% ( 378 / 2499 )

            ----------------Best------------------
            When lbd = 1 FN prefered
            Feature:  ('twoVerticle', 1, 18, 1, 2)
            Training Error: 100.00%
            Training Accuray: 19.93% ( 498 / 2499 )
            Training False Positive: 80.03% ( 2000 / 2499 )
            Training False Negative: 0.04% ( 1 / 2499 )
            '''
            lbd = 0.8
            #error = lbd * fn/len(training_data) + (1- lbd) * fp/len(training_data)

            if error < best_error and error != 0:

                best_threshold, best_feature, best_clf, best_error, best_accuracy, best_fp, best_fn = threshold, feature, classifier, error, accuracy, fp, fn
            
            i += 1 

        return best_threshold, best_feature, best_clf, best_error, best_accuracy, best_fp, best_fn
    
    def apply_features(self, features, training_data):
        """
            Maps features onto the training dataset
        """

        X = np.zeros((len(features), len(training_data)))

        z = [None] * len(features)

        y = np.array(list(map(lambda data: data[1], training_data)))

        print('Applying features to Training set...')

        for i in range(len(training_data)):

            print('Calculating features on data #', i, '/', len(training_data))

            for j in range(len(features)):

                X[j][i] = features[j].compute_feature(training_data[i][0])[1]

                z[j] = features[j].compute_feature(training_data[i][0])[0]
        return X, y, z

    def classify(self, image):
        """
            Classifies an image
            Returns 1 if the image has a face and returns 0 otherwise
        """
        total = 0
        
        integralImg = getIntegralImage(image)
        
        for alpha, clf in zip(self.alphas, self.classifiers):
            
            total += alpha * clf.classify(integralImg)[0]
            
        return 1 if total >= 0.6 * sum(self.alphas) else 0

    def save(self, filename):
        """
            Saves the classifier to a pickle
        """
        with open(filename+".pkl", 'wb') as f:
            
            pickle.dump(self, f)

    @staticmethod
    def Load(filename):
        """
            A static method which loads the classifier from a pickle
        """
        with open(filename+".pkl", 'rb') as f:
            
            return pickle.load(f)

