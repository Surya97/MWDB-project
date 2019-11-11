import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
from tqdm import tqdm
import numpy as np
import misc
import os
from pathlib import Path
from Decomposition import Decomposition
from Metadata import Metadata

class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter=300, test_dataset_path=''):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}
        self.dorsal_features = {}
        self.palmar_features = {}
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                         'Phase2', 'pickle_files')
        self.test_dataset_path = test_dataset_path

    def set_label_features(self):

        if not (os.path.exists(os.path.join(self.reduced_pickle_file_folder,'LBP_PCA.pkl'))):
            print('Pickle file not found for the Particular (model,Reduction)')
            print('Runnning Task1 Of Phase2 for the Particular (model,Reduction) to get the pickle file')
            decomposition = Decomposition('PCA', 30, 'LBP', self.test_dataset_path)
            decomposition.dimensionality_reduction()

        self.dorsal_features = self.get_label_features('dorsal')
        self.palmar_features = self.get_label_features('palmar')


    def get_label_features(self, label):

        if not (os.path.exists(os.path.join(self.reduced_pickle_file_folder, 'LBP_PCA_'+label+'.pkl'))):
            test_dataset_folder_path = os.path.abspath(
                os.path.join(Path(os.getcwd()).parent, self.test_dataset_path))
            images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
            metadata = Metadata(images_list)
            metadata.save_label_decomposed_features(label)

        features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'LBP_PCA_'+label)

        return features

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in tqdm(range(self.max_iter)):
            self.classifications = {}

            for feature in data:
                dists = [np.linalg.norm(feature-self.centroids[centroid]) for centroid in self.centroids]
                classification = dists.index(min(dists))
                if classification in self.classifications:
                    self.classifications[classification].append(feature)
                else:
                    self.classifications[classification] = [feature]

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        dists = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = dists.index(min(dists))
        return classification

