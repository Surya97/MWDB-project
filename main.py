import sys
import cv2
import colorMoments
import glob
import misc
import HOG
import colorMomentSimilarity
import hogSimilarity

args = sys.argv[1:]
tsk = args[0]
path = args[1]
imageId = args[2]
model = args[3]

pathTask1 = path+imageId
pathTask2 = path+'*.jpg'
pathTask3 = pathTask1


if(tsk == '1'):
    for img in glob.glob(pathTask1):
        colorMomentsVector = []
        HOGVector = []
        imgName = misc.getName(img)
        print(imgName)
        
        image=cv2.imread(img)
        colorMomentsVector = colorMoments.findColorMoments(image);
        colorMomentsVector = imgName + colorMomentsVector
        
        misc.saveCSV('task1ColorMomentsData.csv', colorMomentsVector)
        
        HOGVector = HOG.calcHOG(image)
        HOGVector = imgName + HOGVector
        
        misc.saveCSV('task1HogData.csv', HOGVector)


if(tsk == '2'):
    for img in glob.glob(pathTask2):
        colorMomentsVector = []
        HOGVector = []
        
        imgName = misc.getName(img)
        imgName = misc.globGetName(imgName[0])
        image=cv2.imread(img)
        colorMomentsVector = colorMoments.findColorMoments(image);
        colorMomentsVector = imgName + colorMomentsVector
        
        misc.saveCSV('Task2colorMomentsData.csv', colorMomentsVector)
        
        HOGVector = HOG.calcHOG(image)
        HOGVector = imgName + HOGVector
        
        misc.saveCSV('Task2HogData.csv', HOGVector)

if(tsk == '3' and model == "col"):
    for img in glob.glob(pathTask3):
        colorMomentSimilarity.calcColorMomentSiml(img)

if(tsk == '3' and model == "hog"):
    for img in glob.glob(pathTask3):
        hogSimilarity.calcHogSiml(img)