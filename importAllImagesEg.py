import glob
import csv
import cv2 as cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2
import numpy as np
from scipy.stats import skew
import scipy.stats as st
from skimage import feature



fields = ['y_means']


cv_img = []
#for img in glob.glob("D:/Multimedia and web databases/Hands/Hand2/Hand_0010698.jpg"):
for img in glob.glob("D:/Multimedia and web databases/Hands/Hand2/*.jpg"):
    colorMoments = []
    Y_mean = []
    U_mean = []
    V_mean = []
    Y_std = []
    U_std = []
    V_std = []
    Y_skew = []
    U_skew = []
    V_skew = []
    HOGdata = []
    n= cv2.imread(img)
    img=cv2.imread(img)
    
    '''Converting the image from BGR to YUV'''
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
       
    windowsize_r = 100
    windowsize_c = 100
    
    # Crop out the window in sice of 100*100
    for r in range(0,img.shape[0], windowsize_r):
        for c in range(0,img.shape[1], windowsize_c):
            window = img_yuv[r:r+windowsize_r,c:c+windowsize_c]
            
            y, u, v = cv2.split(window)
            y_mean = np.mean(y)
            u_mean = np.mean(u)
            v_mean = np.mean(v)
            
            #print("y_mean", y_mean, "u_mean", u_mean, "v_mean", v_mean)
            Y_mean.append(y_mean)
            U_mean.append(u_mean)
            V_mean.append(v_mean)
            
            y_std = np.std(y)
            u_std = np.std(u)
            v_std = np.std(v)
            
            #print("y_std", y_std, "u_std", u_std, "v_std", v_std)
            Y_std.append(y_std)
            U_std.append(u_std)
            V_std.append(v_std)
            
            '''Convert into np array, flatten it and then calculate skew'''
            y_np = np.array(y)
            y_flatten = y_np.flatten()
            Y_skew = Y_skew+[skew(y_flatten)]
            
            u_np = np.array(u)
            u_flatten = u_np.flatten()
            U_skew = U_skew+[skew(u_flatten)]
            
            v_np = np.array(v)
            v_flatten = v_np.flatten()
            V_skew = V_skew+[skew(v_flatten)]
            
            '''colorMoments.append(y_mean);
            colorMoments.append(u_mean);
            colorMoments.append(v_mean);
            
            colorMoments.append(y_std);
            colorMoments.append(u_std);
            colorMoments.append(v_std);
            
            colorMoments.append(Y_skew);
            colorMoments.append(U_skew);
            colorMoments.append(V_skew);'''
            
            #with open('colorMoments.csv', 'w') as csvFile:

    cv2.destroyAllWindows();
    
    colorMoments = Y_mean+U_mean+V_mean+Y_std+U_std+V_std+Y_skew+U_skew+V_skew
  
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    downGray = cv2.resize(grayImg, (160, 120))    
    H = feature.hog(downGray, orientations=8, pixels_per_cell=(8, 8),
    	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
    
    for i in H:
        colorMomentsAndHog = colorMoments.append(i)    
    
    
    HOGdata.append(H)
    
    with open('hogDataInfo2.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(HOGdata)
        csvFile.close()

    '''WORKING
    with open("myfileY_mean1.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(Y_mean)
    myfile.close()'''
    
    
    with open("myfileY_mean2.csv", 'a', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(colorMoments)
    myfile.close()
    
    cv_img.append(n)
    
