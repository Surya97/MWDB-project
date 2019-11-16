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
from features_images import FeaturesImages

def euclidean_distance(dist1, dist2):
    return (sum([(a-b)**2 for a,b in zip(dist1, dist2)]))**0.5

class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter=300, labeled_dataset_path='', unlabeled_dataset_path= ''):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}
        self.dorsal_features = {}
        self.palmar_features = {}
        self.unlabeled_dataset_path = unlabeled_dataset_path
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        self.labeled_dataset_path = labeled_dataset_path
        self.image_cluster_map = dict()
        self.image_list = []
        self.decomposition = None


    def set_label_features(self):

        if not (os.path.exists(os.path.join(self.reduced_pickle_file_folder, 'LBP_PCA.pkl'))):
            print('Pickle file not found for the Particular (model,Reduction)')
            print('Runnning Task1 Of Phase2 for the Particular (model,Reduction) to get the pickle file')
            self.decomposition = Decomposition('PCA', 30, 'LBP', self.labeled_dataset_path)
            self.decomposition.dimensionality_reduction()
            unlabeled_dataset_features = self.get_unlabeled_images_decomposed_features(self.unlabeled_dataset_path)
            misc.save2pickle(unlabeled_dataset_features, self.reduced_pickle_file_folder,
                             feature=('unlabeled_LBP_PCA'))


        self.dorsal_features = self.get_label_features('dorsal')
        self.palmar_features = self.get_label_features('palmar')

    def get_label_features(self, label):

        if not (os.path.exists(os.path.join(self.reduced_pickle_file_folder, 'LBP_PCA_'+label+'.pkl'))):
            test_dataset_folder_path = os.path.abspath(
                os.path.join(Path(os.getcwd()).parent, self.labeled_dataset_path))
            images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
            metadata = Metadata(images_list)
            metadata.save_label_decomposed_features(label)


        features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'LBP_PCA_'+label)

        return features

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

    def get_similarity_val(self, unlabeled_dataset_path):
       self.fit(self.dorsal_features)
       similarity_value_unlabeledimages={}
       labeled_dataset_features=self.dorsal_features
       unlabeled_dataset_features = misc.load_from_pickle(self.reduced_pickle_file_folder, 'unlabeled_LBP_PCA')
       for unlabeled_image_id, feature in unlabeled_dataset_features.items():
           similarity_val = 0
           count = 0
           dist, unlabeled_cluster_number = self.predict(feature)
           for labeled_image_id, labeled_cluster_number in self.image_cluster_map.items():
                if unlabeled_cluster_number == labeled_cluster_number:
                    similarity_val+=euclidean_distance(feature, labeled_dataset_features[labeled_image_id])
                    count+=1
           similarity_value_unlabeledimages[unlabeled_image_id]=(similarity_val/count)

       return similarity_value_unlabeledimages


    def get_unlabeled_images_decomposed_features(self, unlabeled_dataset_path):
        test_dataset_folder_path = os.path.abspath(
            os.path.join(Path(os.getcwd()).parent, self.labeled_dataset_path))
        images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())

        images_decomposed_features = {}
        for image_id in images_list:
            test_folder_path = os.path.join(Path(os.path.dirname(__file__)).parent, unlabeled_dataset_path)
            features_images = FeaturesImages('LBP', test_folder_path)
            test_image_path = os.path.join(test_folder_path, image_id)
            test_image_features = list()
            test_image_features.append(features_images.compute_image_features(test_image_path))

            unlabeled_image_decomposed_features = self.decomposition.decomposition_model.get_new_image_features_in_latent_space(
                test_image_features)

            images_decomposed_features[image_id] = unlabeled_image_decomposed_features


        return images_decomposed_features