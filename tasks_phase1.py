import sys
import cv2
import my_utils
import featuresextraction
import similarity_functions
# getting all the arguments
args = sys.argv[1:]

# taskid
task = args[0]

# modelnumber
model = args[1]

if task == '1':

    image_id = args[2]
    if model=='cmom':
        images = []
        images.append(cv2.imread('test/' + image_id, 1))
        cmomfeatures_vector = featuresextraction.getcolormoments(images)
        print(cmomfeatures_vector)
    elif model=='sift':
        images = []
        images.append(cv2.imread('test/' + image_id, 1))
        cmomfeatures_vector = featuresextraction.getsift(images)
        print(cmomfeatures_vector)


# given the directory path, extract and store all the features descriptor files in a folder
elif task == '2':
    directory = args[2]
    image_list = my_utils.getImages(directory);
    images_names = my_utils.getImagesName(directory)

    if model == 'cmom':
        cmomfeatures_vector = featuresextraction.getcolormoments(image_list)

        directory_images_features_map = {}
        for i in range(len(image_list)):
            directory_images_features_map[images_names[i]] = cmomfeatures_vector[i]

        my_utils.save2pickle(directory_images_features_map, 'pickleoutput/', 'cmom')
        directory_images_features_map = {}
        directory_images_features_map = my_utils.load_from_picklefile('pickleoutput/', 'cmom')
        print(directory_images_features_map.items())

    elif model=='sift':
        siftfeatures_vector = featuresextraction.getsift(image_list)

        directory_images_features_map = {}
        for i in range(len(image_list)):
            directory_images_features_map[images_names[i]] = siftfeatures_vector[i]

        my_utils.save2pickle(directory_images_features_map, 'pickleoutput/', 'sift')
        directory_images_features_map = {}
        directory_images_features_map = my_utils.load_from_picklefile('pickleoutput/', 'sift')
        print(directory_images_features_map.items())


# given an image ID, a model, and a value “k” returns and visualises k similar images
if task == '3':
    image_id = args[2]
    k = args[3]

    directory_images_features_map = {}
    directory_images_features_map = my_utils.load_from_picklefile('pickleoutput/', 'cmom')


    source_img = cv2.imread('test/' + image_id, 1)


    images = []
    images.append(source_img)
    source_img_vector = featuresextraction.getcolormoments(images)
    if model == 'cmom':
        similarity_functions.cmom_similarity(source_img_vector[0], directory_images_features_map, k)
