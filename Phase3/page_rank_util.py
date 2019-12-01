from PageRank import PageRank
import numpy as np
import collections
from pathlib import Path
import os
import sys
sys.path.insert(1, '../Phase1')
from Decomposition import Decomposition
import misc



def euclidean_distance(image1, image2):
    return (sum([(a - b) ** 2 for a, b in zip(image1, image2)])) ** 0.5


class PageRankUtil:
    def __init__(self, folder_path, k, m, start_images_list=None, alpha=0.85):

        self.image_feature_map = None
        self.images_list = []
        self.pagerank = None
        self.k = k
        self.test_folder_path = folder_path
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        self.start_images_list = start_images_list
        self.image_image_similarity_map = {}
        self.teleportation = []
        self.alpha = alpha
        self.random_walk = []
        self.get_image_dataset_features()
        self.page_ranking = {}
        self.m = m

    def get_image_dataset_features(self):
        decomposition = Decomposition('SVD', k_components=256, feature_extraction_model_name='HOG',
                                      test_folder_path=self.test_folder_path)
        decomposition.dimensionality_reduction()
        self.image_feature_map = misc.load_from_pickle(self.reduced_pickle_file_folder, 'HOG_SVD')
        self.images_list = list(self.image_feature_map.keys())
        self.teleportation = [[0.0 for i in range(1)] for j in range(len(self.images_list))]
        self.random_walk = [[0.0 for i in range(len(self.images_list))] for j in range(len(self.images_list))]
        if self.start_images_list is not None:
            n = len(self.start_images_list)
            for image in self.start_images_list:
                self.teleportation[self.images_list.index(image)][0] = (1-self.alpha)/n

        self.teleportation = np.array(self.teleportation)
        self.create_random_walk()

    def create_random_walk(self):
        for image1, feature1 in self.image_feature_map.items():
            for image2, feature2 in self.image_feature_map.items():
                if image2 != image1:
                    distance = euclidean_distance(feature1, feature2)
                    if image1 in self.image_image_similarity_map:
                        self.image_image_similarity_map[image1].append(tuple((image2, distance)))
                    else:
                        self.image_image_similarity_map[image1] = [tuple((image2, distance))]

        for image, similarity_list in self.image_image_similarity_map.items():
            self.image_image_similarity_map[image] = sorted(self.image_image_similarity_map[image],
                                                            key=lambda x: x[1])[: self.k]

        for image, similarity_list in self.image_image_similarity_map.items():
            image_idx = self.images_list.index(image)
            for image2, distance in similarity_list:
                self.random_walk[image_idx][self.images_list.index(image2)] = distance

        for i in range(len(self.random_walk)):
            temp = self.random_walk[i]
            self.random_walk[i] = [(self.alpha * j)/sum(temp) for j in temp]

        self.random_walk = list(map(list, zip(*self.random_walk)))
        self.random_walk = np.array(self.random_walk)

    def page_rank_util(self):
        pagerank = PageRank(self.random_walk, self.teleportation)
        final_steady_state = pagerank.get_final_steady_state()
        # print(final_steady_state)
        for i in range(len(self.images_list)):
            self.page_ranking[self.images_list[i]] = final_steady_state[i][0]

        # Ordering the page ranking based on the rank
        sorted_pagerank = sorted(self.page_ranking.items(), key=lambda kv: kv[1], reverse=True)
        self.page_ranking = dict(collections.OrderedDict(sorted_pagerank))
        # print(self.page_ranking)

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




