from skimage import feature
import numpy as np
from scipy.stats import itemfreq
from itertools import zip_longest
import misc
import os
import collections

def euclidean_distance(dist1, dist2):
    print(len(dist1), len(dist2))
    return (sum([(a-b)**2 for a,b in zip(dist1, dist2)]))**0.5

def chisquare(dist1, dist2):
    return sum([((a-b)**2/(a+b)) if a+b != 0 else 0 for a,b in zip(dist1, dist2)])


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def compute(self, image):
        return self.compute_lbp(image)

    def compute_lbp(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints, R=self.radius, method='uniform')
        # x = itemfreq(lbp.ravel())
        # hist = x[:, 1]/sum(x[:, 1])
        vecimgLBP = lbp.flatten()
        hist, hist_edges = np.histogram(vecimgLBP, bins=256)
        return hist

    def get_similar_images(self, test_image_feature, k, test_folder_path, test_image):
        dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), 'LBP')
        ranking = {}
        for image_id, feature_vector in dataset_images_features.items():
            distance = chisquare(test_image_feature, feature_vector)
            ranking[image_id] = distance

        sorted_results = collections.OrderedDict(sorted(ranking.items(), key=lambda val: val[1], reverse=False))
        top_k_items = {item: sorted_results[item] for item in list(sorted_results)[:k+1]}

        plot_images = {}
        for image_id in top_k_items.keys():
            if image_id != test_image:
                image_path = os.path.join(test_folder_path, image_id)
                plot_images[image_path] = top_k_items[image_id]

        misc.plot_similar_images(plot_images)


