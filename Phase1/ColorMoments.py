import scipy as sp
import cv2
import misc
import os
import math
import collections


def euclidean_distance(feature_list1, feature_list2):
    return (sum([(a - b) ** 2 for a, b in zip(feature_list1, feature_list2)])) ** 0.5


class ColorMoments:
    def __init__(self):
        self.model_name = 'CM'

    def get_moments_channel(self, channel):
        moment_mean = sp.mean(channel)
        moment_sd = sp.std(channel)
        moment_skew = sp.stats.skew(channel.flatten())
        return [moment_mean, moment_sd, moment_skew]

    def compute(self, image):
        c1, c2, c3 = cv2.split(image)
        image_features_y = self.get_moments_channel(c1)
        image_features_u = self.get_moments_channel(c2)
        image_features_v = self.get_moments_channel(c3)
        features_tuple = [feature for feature in zip(image_features_y, image_features_u, image_features_v)]
        combined_feature = []
        for feature in features_tuple:
            combined_feature.append(feature[0])
            combined_feature.append(feature[1])
            combined_feature.append(feature[2])
        return combined_feature

    def get_similar_images(self, test_image_feature, k, test_folder_path, test_image):
        dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), 'CM')
        ranking = {}
        for image_id, feature_vector in dataset_images_features.items():
            distance = euclidean_distance(test_image_feature, feature_vector)
            ranking[image_id] = distance

        sorted_results = collections.OrderedDict(sorted(ranking.items(), key=lambda val: val[1], reverse=False))
        top_k_items = {item: sorted_results[item] for item in list(sorted_results)[:k+1]}

        plot_images = {}
        for image_id in top_k_items.keys():
            if image_id != test_image:
                image_path = os.path.join(test_folder_path, image_id)
                plot_images[image_path] = top_k_items[image_id]

        misc.plot_similar_images(plot_images)




