import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getName(imageName):
    image_name = []
    imgName = imageName.split('/')
    image_name.append((imgName[len(imgName)-1]))
    return image_name

def globGetName(imageName):
    image_name = []
    imgName = imageName.split('\\')
    image_name.append((imgName[len(imgName)-1]))
    return image_name

def saveCSV(filename, data):
    with open(filename, 'a', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(data)
    myfile.close()
    
def printSimlDict(dict):
    #print(dict)
    for key in dict.keys():
        siml = dict[key]
        key = 'D:/Multimedia and web databases/Hands/Hands/' + key
        fig = plt.figure()
        img=mpimg.imread(key)
        imgplot = plt.imshow(img)
        fig.text(x=0.5, y=0.1, s=siml, verticalalignment='bottom', horizontalalignment='center')
        plt.show()
    
    