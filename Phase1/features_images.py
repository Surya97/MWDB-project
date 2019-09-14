from pathlib import Path
import misc
import os
from tqdm import tqdm
import LBP
import HOG


def print_array(feature_vector):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in feature_vector]))

class FeaturesImages:

    def __init__(self, model_name, folder_path):
        self.model_name = model_name
        self.folder_path = folder_path
        self.split_windows = False
        self.model = None
        if self.model_name == 'LBP':
            self.model = LBP.LocalBinaryPatterns(8, 2)
            self.split_windows = True
        elif self.model_name == 'HOG':
            self.model = HOG.Hog(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

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
            print(len(list(files_in_directory.keys())), len(features_image_folder))
            images = list(files_in_directory.keys())
            folder_images_features_dict = {}
            for i in range(len(images)):
                folder_images_features_dict[images[i]] = features_image_folder[i]

            misc.save2pickle(folder_images_features_dict, os.path.dirname(__file__), feature=self.model_name)

    def compute_image_features(self, image, print_arr=False):
        try:
            image = misc.read_image(os.path.join(os.path.dirname(__file__), image))
            image_gray = misc.convert2gray(image)
            image_feature = []
            if self.split_windows:
                windows = misc.split_into_windows(image_gray, 100, 100)
                for window in windows:
                    window_pattern = self.model.compute(window)
                    if len(image_feature) == 0:
                        image_feature = window_pattern
                    else:
                        image_feature += window_pattern
            else:
                image_feature = self.model.compute(image)
        except OSError as e:
            print(e.strerror)
        finally:
            if not print_arr:
                return image_feature

