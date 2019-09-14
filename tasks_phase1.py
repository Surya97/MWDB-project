import sys
import cv2
import my_utils
import featuresextraction

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
    image_list = my_utils.getImages('test/');
    images_names = my_utils.getImagesName('test/')

    if model == 'cmom':
        cmomfeatures_vector = featuresextraction.getcolormoments(image_list)

        folder_images_features_dict = {}
        for i in range(len(image_list)):
            folder_images_features_dict[images_names[i]] = cmomfeatures_vector[i]

        my_utils.save2pickle(folder_images_features_dict, 'pickleoutput/', 'cmom')
        folder_images_features_dict = {}
        folder_images_features_dict = my_utils.load_from_pickle('pickleoutput/', 'cmom')

    elif model=='sift':
        siftfeatures_vector = featuresextraction.getsift(image_list)

        folder_images_features_dict = {}
        for i in range(len(image_list)):
            folder_images_features_dict[images_names[i]] = siftfeatures_vector[i]

        my_utils.save2pickle(folder_images_features_dict, 'pickleoutput/', 'sift')
        folder_images_features_dict = {}
        folder_images_features_dict = my_utils.load_from_pickle('pickleoutput/', 'sift')



# given an image ID, a model, and a value “k” returns and visualises k similar images
if task == '3':
    image_id = args[2]
    k = args[3]
