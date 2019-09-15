import pickle
import numpy as np
import scipy as sp
from scipy.stats import skew
import cv2
import os
import pandas as pand
import matplotlib.pyplot as matpyplot
import matplotlib.image as mpimg
import math


# get the list of file names in the directory
def getImagesName(directory):
    image_names=[]
    for filename in os.listdir(directory):
        image_names.append(filename)
    return image_names

#get the list of files in directory
def getImages(directory):

    images = []
    for filename in os.listdir(directory):
        images.append(cv2.imread(directory + filename, 1))

    return images

#divide an image in 100*100 pixels window
def getwindows(img):
    windows = []
    rows = img.shape[0]
    columns = img.shape[1]

    for i in range(0, rows, 100):
        for j in range(0, columns, 100):
            windows.append(img[i:i + 100, j:j + 100])

    return windows

# split the image in to 3 channels
def splitchannels(img):
    c1, c2, c3 = cv2.split(img)
    c1_numpy = np.array(c1)
    c2_numpy = np.array(c2)
    c3_numpy = np.array(c3)

    return c1_numpy, c2_numpy, c3_numpy

# return a list of color moments of an image
def getwindowmoments(img_c):
    mean_c = sp.mean(img_c)
    sd_c = sp.std(img_c)
    sk_c = skew(img_c.flatten())

    return [mean_c, sd_c, sk_c]

#save the map in to pickle file
def save2pickle(list, path, model):
        filename = os.path.join(path, model + '.pkl')
        output = open(filename, 'wb')
        pickle.dump(list, output, protocol=2)
        output.close()

#load the map from pickle file
def load_from_picklefile(path, model):
    pickle_file_path = os.path.join(path, model+'.pkl')
    print('pickle file path', pickle_file_path)
    infile = open(pickle_file_path, 'rb')
    dataset_features = pickle.load(infile)

    return dataset_features

#plot each image with its similarity measure value
def plot_similar_images(filename, similarity):

    image = mpimg.imread(os.path.join('test1/', filename))
    fig_plot = matpyplot.figure()
    matpyplot.imshow(image, cmap='Greys_r')
    similarityval_str = 'Value Of Similarity: ' + str(round(similarity, 2))
    fig_plot.text(x=0.7, y=0.4, s=similarityval_str, verticalalignment='bottom', horizontalalignment='center')
    matpyplot.show()


def calc_sift_distance(vec1, vec2):

    len1=len(vec1)
    len2=len(vec2)
    leng = len1 if len1 > len2 else len2

    # one dimensional vector with length 132 each with value 0
    vec = []
    i = 132
    while i > 0:
        vec.append(0)
        i = i - 1

    # adding vector to equate the number of keypoint features
    if leng == len1:
        x = len1-len2
        while x>0:
            vec2.append(vec)
            x = x-1
    else :
        y = len2-len1
        while y>0:
            vec1.append(vec)
            y = y-1

    sum = 0

    for i in range(0, leng):
        for j in range(0,132):
            sum = sum + (vec1[i][j] - vec2[i][j])**2


    return math.sqrt(sum)