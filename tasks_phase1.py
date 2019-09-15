import sys
import cv2
import my_utils
import featuresextraction
import similarity_functions
# getting all the arguments
args = sys.argv[1:]

# taskid
task = args[0]

# modelnumber - which color model to use
model = args[1]

#single image
if task == '1':
    image_id = args[2]
    # colormoments
    if model=='cmom':
        images = []
        images.append(cv2.imread('test/' + image_id, 1))     # reading the image
        cmomfeatures_vector = featuresextraction.getcolormoments(images)   # extracting the features of the image
        print(cmomfeatures_vector)
    # sift
    elif model=='sift':
        images = []
        images.append(cv2.imread('test/' + image_id, 1))     #reading the image
        cmomfeatures_vector = featuresextraction.getsift(images)   #extracting the features of the image
        print(cmomfeatures_vector)


# given the directory path, extract and store all the features descriptor files in a folder
elif task == '2':
    directory = args[2]
    image_list = my_utils.getImages(directory);     # contains all the files names of the directory
    images_names = my_utils.getImagesName(directory)   #contains all the files of the directory

    if model == 'cmom':             # for color moments
        cmomfeatures_vector = featuresextraction.getcolormoments(image_list)  #extracting the color moments of the images

        directory_cmom_features_map = {}   #map - which contains the file_name as the key and feature descriptor as the value
        for i in range(len(image_list)):
            directory_cmom_features_map[images_names[i]] = cmomfeatures_vector[i]     #storing all the images feature descriptors in to a map

        my_utils.save2pickle(directory_cmom_features_map, 'pickleoutput/', 'cmom')    #storing the map in to pickle
        directory_cmom_features_map = {}
        directory_cmom_features_map = my_utils.load_from_picklefile('pickleoutput/', 'cmom')
        print(directory_cmom_features_map.items())                                    #displaying the  content of the pickle file

    elif model=='sift':        # for color moments
        siftfeatures_vector = featuresextraction.getsift(image_list)    #extracting list of vectors for each image in the directory

        directory_sift_features_map = {}                 #map - which contains the file_name as the key and list of vectors as the value for each image
        for i in range(len(image_list)):
            directory_sift_features_map[images_names[i]] = siftfeatures_vector[i]

        my_utils.save2pickle(directory_sift_features_map, 'pickleoutput/', 'sift')    #storing the map in to pickle
        directory_sift_features_map = {}
        directory_sift_features_map = my_utils.load_from_picklefile('pickleoutput/', 'sift')

        for i in directory_sift_features_map:
            print (len(directory_sift_features_map[i]))                     #displaying the  content of the pickle file

# given an image ID, a model, and a value “k” returns and visualises k similar images
if task == '3':
    image_id = args[2]
    k = args[3]

    source_img = cv2.imread('test/' + image_id, 1)


    images = []
    images.append(source_img)

    if model == 'cmom':
        source_cmom_vector = featuresextraction.getcolormoments(images)
        directory_cmom_features_map = {}
        directory_cmom_features_map = my_utils.load_from_picklefile('pickleoutput/', 'cmom')  # loading the cmom features from the pickle file
        similarity_functions.cmom_similarity(source_cmom_vector[0], directory_cmom_features_map, k)

    elif model == 'sift':
        source_sift_vector = featuresextraction.getsift(images)
        directory_sift_features_map = {}
        directory_sift_features_map = my_utils.load_from_picklefile('pickleoutput/','sift')  # loading the sift vectors from the pickle file
        print(len(source_sift_vector[0]))
        similarity_functions.sift_similarity(source_sift_vector[0], directory_sift_features_map, k)