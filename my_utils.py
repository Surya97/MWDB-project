import pickle
import numpy as np
import scipy as sp
from scipy.stats import skew
import cv2
import os
import pandas as pand
import matplotlib.pyplot as matpyplot
import matplotlib.image as mpimg

def getImagesName(directory):
    image_names=[]
    for filename in os.listdir(directory):
        image_names.append(filename)
    return image_names

def getImages(directory):

    images = []
    for filename in os.listdir(directory):
        images.append(cv2.imread(directory + filename, 1))

    return images


def getwindows(img):
    windows = []
    rows = img.shape[0]
    columns = img.shape[1]

    for i in range(0, rows, 100):
        for j in range(0, columns, 100):
            windows.append(img[i:i + 100, j:j + 100])

    return windows


def splitchannels(img):
    c1, c2, c3 = cv2.split(img)
    c1_numpy = np.array(c1)
    c2_numpy = np.array(c2)
    c3_numpy = np.array(c3)

    return c1_numpy, c2_numpy, c3_numpy


def getwindowmoments(img_c):
    mean_c = sp.mean(img_c)
    sd_c = sp.std(img_c)
    sk_c = skew(img_c.flatten())

    return [mean_c, sd_c, sk_c]

def save2csv(tuples, path, feature=''):
    if feature=='cmom':
        path = os.path.join(path, 'cmom.csv')
    elif feature=='sift':
        path = os.path.join(path, 'sift.csv')

    output = pand.DataFrame(tuples, columns=['Image ID', feature])
    output.to_csv(path, index=False)

def save2pickle(tuples, path, feature):
        filename = os.path.join(path, feature + '.pkl')
        outfile = open(filename, 'wb')
        pickle.dump(tuples, outfile, protocol=2)
        outfile.close()

def load_from_picklefile(path, feature):
    final_path = os.path.join(path, feature+'.pkl')
    print('pickle file path', final_path)
    infile = open(final_path, 'rb')
    dataset_features = pickle.load(infile)

    return dataset_features

def plot_similar_images(filename, similarity):

    image = mpimg.imread(os.path.join('test1/', filename))
    fig = matpyplot.figure()

    matpyplot.imshow(image, cmap='Greys_r')
    # plt.axis('off')
    similarity_string = 'Similarity: ' + str(round(similarity, 2))
    fig.text(x=0.5, y=0.1, s=similarity_string, verticalalignment='bottom', horizontalalignment='center')
    matpyplot.show()