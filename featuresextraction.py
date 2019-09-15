import cv2
import my_utils
import numpy as np
import similarity_functions

#return a vector of colormoments for each image
def getcolormoments(image_list):
    imgsdescriptorlistCMOM = []

    for img_bgr in image_list:
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        windows = my_utils.getwindows(img_yuv)    #divides the image in to 192 windows each of 100*100 size
        winds_mom_val_y = []
        winds_mom_val_u = []
        winds_mom_val_v = []

        for wind in windows:
            y_numpy, u_numpy, v_numpy = my_utils.splitchannels(wind)   #splitting the window in to 3 channels y, u, v

            winds_mom_val_y = winds_mom_val_y + (my_utils.getwindowmoments(y_numpy))
            winds_mom_val_u = winds_mom_val_u + (my_utils.getwindowmoments(u_numpy))
            winds_mom_val_v = winds_mom_val_v + (my_utils.getwindowmoments(v_numpy))


        img_mom_val = winds_mom_val_y + winds_mom_val_u + winds_mom_val_v      # moment value of each window

        imgsdescriptorlistCMOM.append(img_mom_val)          #list of vectors which has the colormoments of each image in the directory

    return imgsdescriptorlistCMOM


def getsift(image_list):
    imgsvectorslistSIFT = []
    sift = cv2.xfeatures2d.SIFT_create();

    for img_bgr in image_list:
        img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, image_descriptor = sift.detectAndCompute(img_grey, None)   #this gives the keypoints and descriptor of each keypoint
        k_size = len(keypoints)
        keypointVector = []
        image_final_vector = []
        for ind in range(0, k_size):
            keypointVector.append(keypoints[ind].pt[0])
            keypointVector.append(keypoints[ind].pt[1])
            keypointVector.append(keypoints[ind].size)
            keypointVector.append(keypoints[ind].angle)
            keypointVector += list(image_descriptor[ind])

            image_final_vector.append(keypointVector)           #list of keypoint descriptor vectors for each image
            keypointVector = []

        imgsvectorslistSIFT.append(image_final_vector)

    return imgsvectorslistSIFT    #return the vector of vectors of keypoints of images in the directory
