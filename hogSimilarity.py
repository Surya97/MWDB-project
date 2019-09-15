import cv2
import math
import misc
import HOG
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv

def calcHogSiml(img):
    imgMpimg=mpimg.imread(img)
    imgplot = plt.imshow(imgMpimg)
    fig = plt.figure()
    fig.text(x=0.5, y=0.1, s='IMAGE TO TEST', verticalalignment='bottom', horizontalalignment='center')
    plt.show()
    
    similarity = {}
    imgName = misc.getName(img)
    image=cv2.imread(img)
    HOGVector = HOG.calcHOG(image)
    HOGVector = imgName + HOGVector
    
    with open('HogData.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            cosineSimilarity = cosSimil1(np.asarray(HOGVector[1:], dtype='float64'), np.asarray(row[1:], dtype='float64'))
            #print(row[0],"Cos Siml  = ", cosineSimilarity)
            similarity[row[0]] = cosineSimilarity
            
        sorted_similarity = sorted(similarity.items(), key=lambda kv: kv[1], reverse=True)
        sorted_similarityDict = collections.OrderedDict(sorted_similarity)
        length = 5
        #print(sorted_similarityDict)
        
        topSimilarImagesDict = {length: sorted_similarityDict[length] for length in list(sorted_similarityDict)[:length]}
        print(topSimilarImagesDict)
        misc.printSimlDict(topSimilarImagesDict)
        
def cosSimil(arr1, arr2):
    totXX = 0;
    totYY = 0;
    totXY = 0;
    cosSiml = 0;
    a1_mean = np.mean(arr1)
    a2_mean = np.mean(arr2)
    for i in range(len(arr1)):
        totXX += ((arr1[i]-a1_mean)*(arr1[i]-a1_mean))
    for i in range(len(arr2)):
        totYY += ((arr2[i]-a2_mean)*(arr2[i]-a2_mean))
    for i in range(len(arr1)):
        totXY = totXY + ((arr1[i]-a1_mean)*(arr2[i]-a2_mean))
    totXX = math.sqrt(totXX)
    totYY = math.sqrt(totYY)
    cosSiml = (totXY)/((totXX)*(totYY))
    return cosSiml