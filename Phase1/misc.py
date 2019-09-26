import os
import copy
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import glob
import numpy as np
from skimage.transform import resize
from matplotlib import gridspec
import pickle
from pathlib import Path
from imutils import paths
import argparse
from matplotlib.offsetbox import AnchoredText


def get_images_in_directory(path):
    dirname = os.path.dirname(__file__)
    print(dirname)
    complete_path = os.path.join(dirname, path)
    print("Complete path", complete_path)
    files = {}
    for filename in os.listdir(complete_path):
        files[filename] = os.path.join(complete_path, filename)
    return files


def read_image(image_path, gray=False):
    dirname = os.path.dirname(__file__)
    image = mpimg.imread(os.path.join(dirname, image_path))
    if gray:
        image = convert2gray(image)
    return image


def convert2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def plot_image(image):
    plt.imshow(image, cmap='Greys_r')
    plt.show()

def split_into_windows(image, x, y):
    w, h = image.shape
    windows = []
    for i in range(0, w, 100):
        for j in range(0, h, 100):
            windows.append(image[i:i+100, j:j+100])

    return windows


def resize_image(image, shape):
    resized = resize(image, shape)
    return resized


def save2pickle(tuples, path, feature):
    filename = os.path.join(path, feature+'.pkl')
    outfile = open(filename, 'wb')
    pickle.dump(tuples, outfile, protocol=2)
    outfile.close()


def load_from_pickle(path, feature):
    final_path = os.path.join(path, feature+'.pkl')
    print('pickle file path', final_path)
    infile = open(final_path, 'rb')
    dataset_features = pickle.load(infile)

    return dataset_features


def plot_similar_images(plot_images_dict):
    plots = len(list(plot_images_dict))
    n_cols = 3
    n_rows = int(math.ceil(plots / n_cols))

    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure()
    image_paths = list(plot_images_dict.keys())
    image_similarities = [plot_images_dict[image] for image in image_paths]
    for i in range(plots):
        ax = fig.add_subplot(gs[i])
        image = read_image(image_paths[i])
        im = ax.imshow(image, cmap='Greys_r')
        similarity_string = os.path.basename(image_paths[i]) + '\nSimilarity: ' + str(
                    round(image_similarities[i] * 100, 2))
        ax.axis('off')
        ax.text(x=0.5, y=-0.1, s=similarity_string, verticalalignment='bottom', horizontalalignment='center')
    plt.show()
