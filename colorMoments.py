import cv2
import numpy as np
from scipy.stats import skew
def findColorMoments(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
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
            
            Y_mean.append(y_mean)
            U_mean.append(u_mean)
            V_mean.append(v_mean)
            
            y_std = np.std(y)
            u_std = np.std(u)
            v_std = np.std(v)
            
            Y_std.append(y_std)
            U_std.append(u_std)
            V_std.append(v_std)

            y_np = np.array(y)
            y_flatten = y_np.flatten()
            Y_skew = Y_skew+[skew(y_flatten)]
            
            u_np = np.array(u)
            u_flatten = u_np.flatten()
            U_skew = U_skew+[skew(u_flatten)]
            
            v_np = np.array(v)
            v_flatten = v_np.flatten()
            V_skew = V_skew+[skew(v_flatten)]
            
            #with open('colorMoments.csv', 'w') as csvFile:

    cv2.destroyAllWindows();
    colorMoments = Y_mean+U_mean+V_mean+Y_std+U_std+V_std+Y_skew+U_skew+V_skew
    
    return colorMoments;