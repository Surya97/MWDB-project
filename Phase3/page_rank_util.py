from PageRank import PageRank
import numpy as np
import collections
from pathlib import Path
import os
import sys
sys.path.insert(1, '../Phase1')
import misc
from features_images import FeaturesImages
import copy
import numpy as np


def euclidean_distance(image1, image2):
    dist1 = np.array(image1)
    dist2 = np.array(image2)
    return np.linalg.norm(dist1 - dist2)


class PageRankUtil:
    def __init__(self, folder_path, k, m, start_images_list=None, alpha=0.85, unlabelled_image=None,
                 image_list=None, feature_map=None, feature_name='HOG'):

        self.image_feature_map = None
        self.images_list = []
        self.pagerank = None
        self.k = k
        self.test_folder_path = folder_path
        self.pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent, 'Phase1')
        self.start_images_list = start_images_list
        self.image_image_similarity_map = {}
        self.unlabelled_image = unlabelled_image
        self.temp_image_list = image_list
        self.temp_feature_map = feature_map
        self.original_feature_map = {}
        self.feature_name = feature_name
        self.original_image_list = []
        self.teleportation = []
        self.alpha = alpha
        self.random_walk = []
        if self.unlabelled_image is None and self.test_folder_path != '':
            self.get_image_dataset_features()
        self.initialize()
        self.page_ranking = {}
        self.m = m

    def get_image_dataset_features(self):
        features_obj = FeaturesImages(self.feature_name, self.test_folder_path)
        features_obj.compute_features_images_folder()
        self.image_feature_map = misc.load_from_pickle(self.pickle_file_folder, self.feature_name)
        self.images_list = list(self.image_feature_map.keys())
        self.original_feature_map = copy.deepcopy(self.image_feature_map)
        self.original_image_list = copy.deepcopy(self.images_list)

    def get_unlabelled_classification_image_features(self, image_id, unlabelled_folder_path):
        test_image_path = os.path.join(Path(os.path.dirname(__file__)).parent, unlabelled_folder_path, image_id)
        features_images = FeaturesImages(self.feature_name,
                                         os.path.join(Path(os.path.dirname(__file__)).parent, unlabelled_folder_path))
        unlabelled_image_features = features_images.compute_image_features(test_image_path)
        return [unlabelled_image_features]

    def initialize(self):
        # print('Initializing random walk and teleportation matrices.........')
        if self.image_feature_map is None:
            # print('Image map is None')
            self.set_image_list_and_feature_map(self.temp_image_list, self.temp_feature_map)
        if self.unlabelled_image is not None:
            # print('Adding unlabeled image')
            unlabelled_image = list(self.unlabelled_image.keys())[0]
            self.images_list.append(unlabelled_image)
            if type(self.unlabelled_image[unlabelled_image]) != list:
                unlabelled_image_feature = self.get_unlabelled_classification_image_features(
                    image_id=unlabelled_image, unlabelled_folder_path=self.unlabelled_image[unlabelled_image])
            else:
                unlabelled_image_feature = self.unlabelled_image[unlabelled_image]
            self.image_feature_map[unlabelled_image] = unlabelled_image_feature

        self.teleportation = [[0.0 for i in range(1)] for j in range(len(self.images_list))]
        self.random_walk = [[0.0 for i in range(len(self.images_list))] for j in range(len(self.images_list))]
        # print('start images list', self.start_images_list)
        if self.start_images_list is not None:
            n = len(self.start_images_list)
            for image in self.start_images_list:
                # print('start image', image)
                self.teleportation[self.images_list.index(image)][0] = (1-self.alpha)/n

        self.teleportation = np.array(self.teleportation)
        self.create_random_walk()

    def create_random_walk(self):
        # print('Creating Random Walk using image image similarities..............')
        for image1, feature1 in self.image_feature_map.items():
            for image2, feature2 in self.image_feature_map.items():
                if image2 != image1:
                    # print('Image1', image1, 'feature1', feature1)
                    # print('Image2', image2, 'feature2', feature2)
                    distance = euclidean_distance(feature1, feature2)
                    if image1 in self.image_image_similarity_map:
                        self.image_image_similarity_map[image1].append(tuple((image2, distance)))
                    else:
                        self.image_image_similarity_map[image1] = [tuple((image2, distance))]

        for image, similarity_list in self.image_image_similarity_map.items():
            self.image_image_similarity_map[image] = sorted(self.image_image_similarity_map[image],
                                                            key=lambda x: x[1])[: self.k]
        # print('Image image similarity')
        # print(self.image_image_similarity_map)

        for image, similarity_list in self.image_image_similarity_map.items():
            image_idx = self.images_list.index(image)
            for image2, distance in similarity_list:
                self.random_walk[image_idx][self.images_list.index(image2)] = distance
        # print('random_walk')
        # print(self.random_walk)
        for i in range(len(self.random_walk)):
            temp = self.random_walk[i]
            self.random_walk[i] = [(self.alpha * j)/sum(temp) for j in temp]

        self.random_walk = list(map(list, zip(*self.random_walk)))
        self.random_walk = np.array(self.random_walk)

    def page_rank_util(self):
        # print('Starting Page rank...........')
        pagerank = PageRank(self.random_walk, self.teleportation)
        final_steady_state = pagerank.get_final_steady_state()
        # print(final_steady_state)
        for i in range(len(self.images_list)):
            self.page_ranking[self.images_list[i]] = final_steady_state[i][0]

        # Ordering the page ranking based on the rank
        sorted_pagerank = sorted(self.page_ranking.items(), key=lambda kv: kv[1], reverse=True)
        self.page_ranking = dict(collections.OrderedDict(sorted_pagerank))
        # print(self.page_ranking)

    def get_page_ranking(self):
        return self.page_ranking

    def set_unlabelled_image(self, unlabelled_image):
        self.unlabelled_image = unlabelled_image

    def set_start_images_list(self, image_list):
        if type(image_list) != list:
            self.start_images_list = [image_list]
        else:
            self.start_images_list = image_list

    def get_decomposition(self):
        return self.decomposition

    def get_original_image_feature_map(self):
        return self.original_feature_map

    def get_image_list_and_feature_map(self):
        return self.images_list, self.image_feature_map

    def get_original_image_list(self):
        return self.original_image_list

    def set_image_list_and_feature_map(self, image_list, image_feature_map):
        self.images_list = image_list
        self.image_feature_map = image_feature_map

    def plot_k_similar(self):
        count = 1
        m_image_rank_map = {}
        for image, rank in self.page_ranking.items():
            image_path = os.path.join(Path(os.path.dirname(__file__)).parent, self.test_folder_path, image)
            m_image_rank_map[image_path] = rank
            count += 1
            if count > self.m:
                break
        # print(m_image_rank_map)
        misc.plot_similar_images(m_image_rank_map, text='\nRanking: ')




