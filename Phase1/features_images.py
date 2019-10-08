from pathlib import Path
from Phase1 import misc
import os
from tqdm import tqdm
from Phase1 import LBP
from Phase1 import HOG
from Phase1 import ColorMoments
from Phase1 import SIFT
import sys


'''
Class for handling all the feature vector generation for both single and multiple images
'''


class FeaturesImages:

    # Initialize the class variables model name, folder path and model
    def __init__(self, model_name, folder_path=None):
        self.model_name = model_name
        self.folder_path = folder_path
        self.split_windows = False
        self.model = None
        if self.model_name == 'LBP':
            self.model = LBP.LocalBinaryPatterns(8, 2)
            self.split_windows = True
        elif self.model_name == 'HOG':
            self.model = HOG.Hog(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        elif self.model_name == 'CM':
            self.model = ColorMoments.ColorMoments()
            self.split_windows = True
        elif self.model_name == 'SIFT':
            self.model = SIFT.SIFT()

    '''
    A getter function to get the initialised feature extraction model.
    '''
    def get_model(self):
        return self.model

    '''
    For the folder path specified, the below function computes the feature vectors
    and stores them in a pickle file based on the respective feature extraction model using
    a helper function 'compute_image_features'. Use of the package 'tqdm' shows a 
    smart progress meter.
    '''

    def compute_features_images_folder(self):
        if self.model is None:
            raise Exception("No model is defined")
        else:
            folder = os.path.join(Path(os.path.dirname(__file__)).parent, self.folder_path)
            files_in_directory = misc.get_images_in_directory(folder)
            features_image_folder = []
            for file, path in tqdm(files_in_directory.items()):
                image_feature = self.compute_image_features(path, print_arr=False)
                features_image_folder.append(image_feature)
            images = list(files_in_directory.keys())
            folder_images_features_dict = {}
            for i in range(len(images)):
                folder_images_features_dict[images[i]] = features_image_folder[i]

            misc.save2pickle(folder_images_features_dict, os.path.dirname(__file__), feature=self.model_name)

    '''
    Given an image path, based on the model requirements the features vectors are retrieved and
    based on the attribute value for 'print_arr' the function either prints the feature vector 
    or returns the feature vector.
    '''

    def compute_image_features(self, image, print_arr=False):
        image_feature = []
        try:
            image_path = os.path.join(os.path.dirname(__file__), image)
            image = misc.read_image(image_path)
            converted_image = misc.convert2gray(image)
            if self.model_name == 'CM':
                converted_image = misc.convert2yuv(image)
            if self.model_name == 'HOG':
                converted_image = misc.resize_image(converted_image, (120, 160))
            if self.split_windows:
                windows = misc.split_into_windows(converted_image, 100, 100)
                for window in windows:
                    window_pattern = self.model.compute(window)
                    if len(image_feature) == 0:
                        image_feature = window_pattern
                    else:
                        image_feature += window_pattern
                # print(len(image_feature))
            else:
                image_feature = self.model.compute(converted_image)
        except OSError as e:
            print("Features_image", e.strerror)
            sys.exit()
        finally:
            if not print_arr:
                return image_feature
            else:
                print(image_feature)
