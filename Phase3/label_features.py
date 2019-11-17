import os
import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
from Decomposition import Decomposition
from pathlib import Path
from Metadata import Metadata
import misc
from features_images import FeaturesImages


class LabelFeatures:
    def __init__(self, labeled_dataset_path='', unlabeled_dataset_path=''):
        self.labeled_dataset_path = labeled_dataset_path
        self.unlabeled_dataset_path = unlabeled_dataset_path
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        self.dorsal_features = None
        self.palmar_features = None
        self.decomposition = None
        self.unlabeled_dataset_features = None

    def get_unlabeled_dataset_features(self):
        self.unlabeled_dataset_features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'unlabeled_LBP_PCA')
        return self.unlabeled_dataset_features

    def set_features(self):
        if not (os.path.exists(os.path.join(self.reduced_pickle_file_folder, 'LBP_PCA.pkl'))):
            print('Pickle file not available. Getting features....')
            self.decomposition = Decomposition('PCA', 40, 'LBP', self.labeled_dataset_path)
            self.decomposition.dimensionality_reduction()
            self.unlabeled_dataset_features = self.get_unlabeled_images_decomposed_features(self.unlabeled_dataset_path)
            misc.save2pickle(self.unlabeled_dataset_features, self.reduced_pickle_file_folder,
                             feature='unlabeled_LBP_PCA')
        self.dorsal_features = self.get_features('dorsal')
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
        if not (os.path.exists(os.path.join(self.reduced_pickle_file_folder, 'LBP_PCA_' + label + '.pkl'))):
            test_dataset_folder_path = os.path.abspath(
                os.path.join(Path(os.getcwd()).parent, self.labeled_dataset_path))
            images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
            metadata = Metadata(images_list)
            metadata.save_label_decomposed_features(label)

        features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'LBP_PCA_'+label)
        return features

    def get_unlabeled_images_decomposed_features(self):
        test_dataset_folder_path = os.path.abspath(
            os.path.join(Path(os.getcwd()).parent, self.unlabeled_dataset_path))
        images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
        images_decomposed_features = {}
        for image_id in images_list:
            features_images = FeaturesImages('LBP', test_dataset_folder_path)
            test_image_path = os.path.join(test_dataset_folder_path, image_id)
            test_image_features = list()
            test_image_features.append(features_images.compute_image_features(test_image_path))

            decomposed_features = self.decomposition.decomposition_model.get_new_image_features_in_latent_space(
                test_image_features)

            images_decomposed_features[image_id] = decomposed_features

        return images_decomposed_features

