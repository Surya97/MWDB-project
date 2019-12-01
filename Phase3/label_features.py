import os
import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
from Decomposition import Decomposition
from pathlib import Path
from Metadata import Metadata
import misc
from features_images import FeaturesImages
import numpy as np


class LabelFeatures:
    def __init__(self, labelled_dataset_path='', unlabelled_dataset_path='', feature_name='HOG',
                 decomposition_name='SVD'):
        self.labelled_dataset_path = labelled_dataset_path
        self.unlabelled_dataset_path = unlabelled_dataset_path
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        self.main_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent, 'Phase1')
        self.dorsal_features = None
        self.palmar_features = None
        self.decomposition = None
        self.unlabelled_dataset_features = None
        self.feature_name = feature_name
        self.decomposition_name = decomposition_name
        self.decomposed_feature = self.feature_name + "_" + self.decomposition_name

    def get_unlabelled_dataset_features(self):
        self.unlabelled_dataset_features = misc.load_from_pickle(self.reduced_pickle_file_folder,
                                                                 'unlabelled_' + self.decomposed_feature)
        return self.unlabelled_dataset_features

    def set_features(self):
        if self.decomposition_name != '':
            self.decomposition = Decomposition(self.decomposition_name, 100, self.feature_name,
                                               self.labelled_dataset_path)
            self.decomposition.dimensionality_reduction()
        else:
            test_dataset_folder_path = os.path.abspath(
                os.path.join(Path(os.getcwd()).parent, self.labelled_dataset_path))
            print('Getting the Model Features from Phase1')
            features_obj = FeaturesImages(self.feature_name, test_dataset_folder_path)
            features_obj.compute_features_images_folder()
        self.unlabelled_dataset_features = self.get_unlabelled_images_decomposed_features()
        misc.save2pickle(self.unlabelled_dataset_features, self.reduced_pickle_file_folder,
                         feature='unlabelled_' + self.decomposed_feature)
        print("Getting features for dorsal_images ")
        self.dorsal_features = self.get_features('dorsal')
        print("Getting features for palmar images")
        self.palmar_features = self.get_features('palmar')

    def get_label_features(self, label):
        if label == "dorsal":
            if self.dorsal_features is None:
                self.set_features()
            return self.dorsal_features
        elif label == "palmar":
            if self.palmar_features is None:
                self.set_features()
            return self.palmar_features

    def get_features(self, label):
        test_dataset_folder_path = os.path.abspath(
            os.path.join(Path(os.getcwd()).parent, self.labelled_dataset_path))
        images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
        metadata = Metadata(images_list)
        if self.feature_name != 'SIFT':
            metadata.save_label_decomposed_features(label, self.decomposed_feature)
            features = misc.load_from_pickle(self.reduced_pickle_file_folder, self.decomposed_feature + '_' + label)
        else:
            features = {}
            database_features = misc.load_from_pickle(self.main_pickle_file_folder, self.feature_name)
            label_images_list = metadata.get_specific_metadata_images_list(feature_dict={'aspectOfHand': label})
            for image in label_images_list:
                features[image] = database_features[image]
        return features

    def get_unlabelled_images_decomposed_features(self):
        test_dataset_folder_path = os.path.abspath(
            os.path.join(Path(os.getcwd()).parent, self.unlabelled_dataset_path))
        images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
        images_decomposed_features = {}

        for image_id in images_list:
            features_images = FeaturesImages(self.feature_name, test_dataset_folder_path)
            test_image_path = os.path.join(test_dataset_folder_path, image_id)
            test_image_features = list()
            test_image_features.append(features_images.compute_image_features(test_image_path))
            if self.decomposition_name != '':
                decomposed_features = self.decomposition.decomposition_model.get_new_image_features_in_latent_space(
                    test_image_features)
                images_decomposed_features[image_id] = decomposed_features
            else:
                images_decomposed_features[image_id] = test_image_features

        return images_decomposed_features


