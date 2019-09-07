import os
import copy
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from imutils import paths
import argparse


def read_image(image_path, gray=False):
    print(os.getcwd())
    dirname = os.path.dirname(__file__)
    image = mpimg.imread(os.path.join(dirname, image_path))
    if gray:
        image = convert2gray(image)
    return image


def convert2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_image(gray)
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

