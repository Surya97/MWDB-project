import cv2
from skimage import feature
    
def calcHOG(img):
    HOGdata = []
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    downGray = cv2.resize(grayImg, (160, 120))    
    H = feature.hog(downGray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2))
        
    for i in H:
        HOGdata.append(i)
        
    return HOGdata;