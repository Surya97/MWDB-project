import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
from tqdm import tqdm
import numpy as np


def euclidean_distance(dist1, dist2):
    return (sum([(a-b)**2 for a, b in zip(dist1, dist2)]))**0.5


class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}
        self.image_cluster_map = dict()
        self.image_list = []

    def fit(self, data):
        self.centroids = {}

        self.image_list = list(data.keys())
        features = list(data.values())
        self.image_cluster_map = dict()

        for i in range(self.k):
            self.centroids[i] = features[i]
            self.image_cluster_map[self.image_list[i]] = i

        for i in tqdm(range(self.max_iter)):
            print('iteration', i)
            self.classifications = {}

            for j in range(len(features)):
                feature = features[j]
                dists = [np.linalg.norm(feature-self.centroids[centroid]) for centroid in self.centroids]
                classification = dists.index(min(dists))
                if classification in self.classifications:
                    self.classifications[classification].append(feature)
                else:
                    self.classifications[classification] = [feature]
                self.image_cluster_map[self.image_list[j]] = classification

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
                print(self.image_cluster_map)
                break

    def predict(self, data):
        dists = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = dists.index(min(dists))
        return min(dists), classification

    def get_image_cluster_map(self):
        return self.image_cluster_map

    def get_similarity_val(self, labelled_dataset_features, unlabelled_dataset_features):
        unlabelled_images_similarity = {}
        for unlabelled_image_id, feature in unlabelled_dataset_features.items():
            similarity_val = 0
            count = 0
            dist, unlabelled_cluster_number = self.predict(feature)
            for labelled_image_id, labelled_cluster_number in self.image_cluster_map.items():
                if unlabelled_cluster_number == labelled_cluster_number:
                    similarity_val += euclidean_distance(feature, labelled_dataset_features[labelled_image_id])
                    count += 1
            unlabelled_images_similarity[unlabelled_image_id] = (similarity_val/count)
        return unlabelled_images_similarity
