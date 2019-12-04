import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
import numpy as np


def euclidean_distance(dist1, dist2):
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    return np.linalg.norm(dist1-dist2)


class KMeans:
    def __init__(self, k, tolerance=0.000001, max_iter=300):
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
        # print(type(features[0]), type(self.centroids[0]))
        for i in range(self.max_iter):
            print('Iteration', i)
            self.classifications = {}

            for _ in range(self.k):
                self.classifications[_] = []

            for j in range(len(features)):
                feature = features[j]
                dists = [np.linalg.norm(np.array(feature)-np.array(self.centroids[centroid])) for centroid in self.centroids]
                classification = dists.index(min(dists))
                self.classifications[classification].append(feature)
                self.image_cluster_map[self.image_list[j]] = classification

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                # print(c)
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # print("Error", np.sum((current_centroid - original_centroid) / (original_centroid * 100.0)))
                # if abs(np.sum((current_centroid - original_centroid) / (original_centroid * 100.0))) > self.tolerance:
                #     print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                #     optimized = False
                if not np.all(original_centroid == current_centroid):
                    optimized = False

            if optimized:
                print(self.image_cluster_map)
                print()
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
                    similarity_val += euclidean_distance(list(feature),
                                                         labelled_dataset_features[labelled_image_id])
                    count += 1
            unlabelled_images_similarity[unlabelled_image_id] = (similarity_val/count)
        return unlabelled_images_similarity
