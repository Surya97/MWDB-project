import sys
sys.path.insert(1, '../Phase1')
from features_images import FeaturesImages
import misc
import os
from pathlib import Path
from PCA import PCAModel
from SVD import SVD
from NMF import NMFModel
from LDA import LDAModel


class Decomposition:
    def __init__(self, decomposition_name, k_components, feature_extraction_model_name):
        self.decomposition_name = decomposition_name
        self.k_components = k_components
        self.decomposition_model = None
        self.feature_extraction_model_name = feature_extraction_model_name
        self.feature_extraction_model = FeaturesImages(self.feature_extraction_model_name)
        self.database_matrix = []
        self.set_database_matrix()

    def set_database_matrix(self):
        parent_directory_path = Path(os.path.dirname(__file__)).parent
        pickle_file_directory = os.path.join(parent_directory_path, 'Phase1')
        print('pickle file directory', pickle_file_directory)
        database_images_features = misc.load_from_pickle(pickle_file_directory, self.feature_extraction_model_name)
        for image_id, feature_vector in database_images_features.items():
            self.database_matrix.append(feature_vector)

    def dimensionality_reduction(self):
        if self.decomposition_name == 'PCA':
            self.decomposition_model = PCAModel(self.database_matrix, self.k_components)
        elif self.decomposition_name == 'SVD':
            self.decomposition_model = SVD(self.database_matrix, self.k_components)
        elif self.decomposition_name == 'NMF':
            if self.feature_extraction_model_name=='CM':
                print('CM is not feasible for NMF Decomposition')
            else:
                self.decomposition_model = NMFModel(self.database_matrix, self.k_components)
        elif self.decomposition_name == 'LDA':
            if self.feature_extraction_model_name=='CM':
                print('CM is not feasible for LDA Decomposition')
            else:
                self.decomposition_model = LDAModel(self.database_matrix, self.k_components)


        self.decomposition_model.decompose()

