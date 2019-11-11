import sys
sys.path.insert(1, '../Phase1')
from tqdm import tqdm
import numpy as np
import misc
import os
from Path import Path


class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}
        self.dorsal_features = {}
        self.palmer_features = {}
        self.reduced_pickle_file_folder = os.path.join(os.path.dirname(__file__), 'pickle_files')

    def get_features_pickle_file(self):
        self.dorsal_features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'LBP_PCA')
        self.palmer_features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'LBP_PCA')

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

