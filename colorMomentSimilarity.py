import cv2
import colorMoments
import misc
import csv
import math
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calcColorMomentSiml(img):
    
    imgMpimg=mpimg.imread(img)
    imgplot = plt.imshow(imgMpimg)
    fig = plt.figure()
    fig.text(x=0.5, y=0.1, s='IMAGE TO TEST', verticalalignment='bottom', horizontalalignment='center')
    plt.show()
    
    similarity = {}
    imgName = misc.getName(img)
    image=cv2.imread(img)
    colorMomentsVector = colorMoments.findColorMoments(image);
    colorMomentsVector = imgName + colorMomentsVector
    
    with open('colorMomentsData.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            #D = distance.euclidean(np.asarray(colorMoments[1:], dtype='float64'), np.asarray(row[1:], dtype='float64'))
            D = math.sqrt(sum([(a - b) ** 2 for a, b in zip((np.asarray(colorMomentsVector[1:], dtype='float64')), (np.asarray(row[1:], dtype='float64')))]))
            #print(row[0],"Distance Measure = ", D)
            similarity[row[0]] = D
            
        sorted_similarity = sorted(similarity.items(), key=lambda kv: kv[1])
        sorted_similarityDict = collections.OrderedDict(sorted_similarity)
        length = 5
        #print(sorted_similarityDict)
        
        topSimilarImagesDict = {length: sorted_similarityDict[length] for length in list(sorted_similarityDict)[:length]}
        misc.printSimlDict(topSimilarImagesDict)
        
        