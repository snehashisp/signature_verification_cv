#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import inspect
import random
import os

#return keypoints from an image using SURF
def getSURFMatches(image,param):

    #use SURF to get set of keypoints and corresponding descriptors
    sift = cv2.xfeatures2d.SURF_create(param)
    kp1,des1 = sift.detectAndCompute(image,None)
    return kp1,des1

#returns set of keypoints from the set of images provided
def getRefSet(imageSet,param):
    
    pointsSet = []
    for img in imageSet:
        p,d = getSURFMatches(img,param)
        pointsSet += [d]
    return pointsSet

#returns a set of images present in a folder
def getImageSet(foldername):
    images = [cv2.imread(foldername + str(i)) for i in os.listdir(foldername)]
    return images

#computes the set of points in a query image that are less than an average distance from 
#the set of reference points
def getBestPoints(rpset,qpset):
    
    dists = []
    for q in qpset:
        sumn = np.sum(np.linalg.norm(rpset - q))
        dists += [sumn]
    
    avg = sum(dists)/len(dists)
    points = []
    for i,d in enumerate(dists):
        if d < avg:
            points += [qpset[i]]
            
    return points

#computes the set of stable points for as set of images using n-1 cross validation methods
def createTestSet(imageSet,param):
    
    #get total set of SURF points for all images
    totalPset = getRefSet(imageSet,param)
    tsetPoints = []
    for i,qpset in enumerate(totalPset):
        #reference set
        rpset = np.concatenate(totalPset[0:i] + totalPset[i+1:])
        tsetPoints += [getBestPoints(rpset,qpset)]
        
    return tsetPoints
        


#imageset = getImageSet("Dataset_4NSigComp2010/TrainingSet/Reference/")
#rset = createTestSet(imageset,400)
#nrset = np.concatenate(rset)
#nrset.shape

