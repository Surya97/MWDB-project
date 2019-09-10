import os
import copy
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import glob
import numpy as np
from skimage.transform import resize
from imutils import paths
import argparse


def get_images_in_directory(path):
    dirname = os.path.dirname(__file__)
    print(dirname)
    complete_path = os.path.join(dirname, path)
    print("Complete path", complete_path)
    files = {}
    for filename in os.listdir(complete_path):
        # print("File", filename)
        files[filename] = os.path.join(complete_path, filename)
    return files


def read_image(image_path, gray=False):
    # print(os.getcwd())
    dirname = os.path.dirname(__file__)
    image = mpimg.imread(os.path.join(dirname, image_path))
    if gray:
        image = convert2gray(image)
    return image


def convert2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plot_image(gray)
    return gray


def plot_image(image):
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def show_grid(image, x, y):
    dx = x
    dy = y
    grid_color = (255, 0, 0)
    image_grid = copy.deepcopy(image)
    image_grid[:, ::dy, :] = grid_color
    image_grid[::dx, :, :] = grid_color
    plot_image(image_grid)


def split_into_windows(image, x, y):
    w, h = image.shape
    windows = image.reshape(w//x, x, -1, y)\
        .swapaxes(1, 2)\
        .reshape(-1, x, y)

    return windows


def combine_into_image(arr, x, y):
    arr = np.array(arr)
    n, rows, cols = arr.shape
    image = arr.reshape(x//rows, -1, rows, cols)\
        .swapaxes(1, 2)\
        .reshape(x, y)
    print(image)
    return image


def resize_image(image, shape):
    resized = resize(image, shape)
    return resized


def save2csv(images, patterns, path, feature=''):
    path = os.path.join(path, 'LBP.csv')
    output = pd.DataFrame({'Image ID': images, 'LBP': patterns})
    output.to_csv(path, index=False)
