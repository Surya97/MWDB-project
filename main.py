import cv2
import colorMoments
import glob
import misc
import HOG
import colorMomentSimilarity

for img in glob.glob("D:/Multimedia and web databases/Hands/Hand2/*.jpg"):
    colorMomentsVector = []
    HOGVector = []
    
    imgName = misc.getName(img)
    
    image=cv2.imread(img)
    colorMomentsVector = colorMoments.findColorMoments(image);
    colorMomentsVector = imgName + colorMomentsVector
    
    misc.saveCSV('colorMomentsData.csv', colorMomentsVector)
    
    HOGVector = HOG.calcHOG(image)
    HOGVector = imgName + HOGVector
    
    misc.saveCSV('HogData.csv', HOGVector)
    
for img in glob.glob("D:/Multimedia and web databases/Hands/Hands/Hand_0001072 - Copy.jpg"):
    colorMomentSimilarity.calcColorMomentSiml(img)
    
    