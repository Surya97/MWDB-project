import cv2
import my_utils
import numpy as np
import similarity_functions
import featuresextraction



if __name__ == "__main__":
    image_list = my_utils.getImages('test/');

    # Compute the features and store the features into respective .csv files

    # Color Moments
    images_names=my_utils.getImagesName('test/')
    cmomfeatures_vector = featuresextraction.getcolormoments(image_list)
    folder_images_features_dict = {}
    for i in range(len(image_list)):
        folder_images_features_dict[images_names[i]] = cmomfeatures_vector[i]


    #my_utils.save2csv(folder_images_features_dict, 'output/', cmomfeatures_vector)
    my_utils.save2pickle(folder_images_features_dict, 'pickleoutput/', 'cmom')
    folder_images_features_dict = {}
    folder_images_features_dict = my_utils.load_from_pickle('pickleoutput/', 'cmom' )

    print(folder_images_features_dict.items())