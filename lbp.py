import cv2
import matplotlib.pyplot as plt
import skimage.feature as ft
img = cv2.imread('/Users/shankar/Desktop/multimedia515/project/Hands/Hand_0000002.jpg' ,0)

windowsize_r = 100
windowsize_c = 100

for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
        window = img[r:r+windowsize_r,c:c+windowsize_c]
        ft.local_binary_pattern(img, p, radius, method= 'default')
        #cv2.imshow('image',window)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
