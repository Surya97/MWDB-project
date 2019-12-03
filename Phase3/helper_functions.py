import os
import sys
sys.path.insert(1, '../Phase1')
from pathlib import Path
import misc
from features_images import FeaturesImages


def get_images_list(folder_path):
    folder = os.path.join(Path(os.path.dirname(__file__)).parent, folder_path)
    return misc.get_images_in_directory(folder)


def get_main_features(feature_name, dataset_folder_path):
    folder = os.path.join(Path(os.path.dirname(__file__)).parent, 'Phase1')
    feature_extraction_object = FeaturesImages(feature_name, dataset_folder_path)
    feature_extraction_object.compute_features_images_folder()
    return misc.load_from_pickle(folder, feature_name)
