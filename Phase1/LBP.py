from skimage import feature
import numpy as np
from scipy.stats import itemfreq
from itertools import zip_longest
import misc
import os
import collections


class SpearmanRanking:
    def __init__(self, test_feature):
        self.test_feature = test_feature
        self.test_feature_rank_vector = self.ranking(self.test_feature)

    def ranking(self, ar):
        n = len(ar)
        rank_vector = [0] * n
        for i in range(n):
            l = 1
            m = 1
            for j in range(i):
                if ar[j] < ar[i]:
                    l += 1
                if ar[j] == ar[i]:
                    m += 1

            for j in range(i + 1, n):
                if ar[j] < ar[i]:
                    l += 1
                if ar[j] == ar[i]:
                    m += 1

            rank_vector[i] = l + (m - 1) * 0.5

        return rank_vector

    def correlation_coefficient(self, y):
        x = self.test_feature_rank_vector
        sum_x = sum(x)
        y = self.ranking(y)
        sum_y = sum(y)
        fill_value = 0.0
        x_median = np.median(x)
        y_median = np.median(y)

        if isinstance(x_median, list) and len(x_median) > 1:
            x_median = np.mean(x_median)
        if isinstance(y_median, list) and len(y_median) > 1:
            y_median = np.mean(y_median)

        if len(x) < len(y):
            fill_value = x_median
        else:
            fill_value = y_median

        diff_array = [a - b for a, b in zip_longest(x, y, fillvalue=fill_value)]
        n = len(diff_array)
        coefficient = 1 - (sum([a * a for a in diff_array])) / (n * (n ** 2 - 1))

        return coefficient


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def compute(self, image):
        return self.compute_lbp(image)

    def compute_lbp(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints, R=self.radius, method='uniform')
        # (hist, _) = np.histogram(lbp.ravel(),
        #                          bins=200,
        #                          range=(0.0, 255.0))
        x = itemfreq(lbp.ravel())

        hist = x[:, 1]/sum(x[:, 1])

        # print('Len hist to list', len(hist.tolist()))
        # return the histogram of Local Binary Patterns
        return hist.tolist()

    def get_similar_images(self, test_image_feature, k, test_folder_path, test_image):
        # misc.plot_image(misc.read_image(os.path.join(test_folder_path, test_image)))
        spearman_similarity = SpearmanRanking(test_image_feature)
        dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), 'LBP')
        ranking = {}
        for image_id, feature_vector in dataset_images_features.items():
            image_feature_rank = spearman_similarity.ranking(feature_vector)
            correlation = spearman_similarity.correlation_coefficient(image_feature_rank)
            ranking[image_id] = correlation

        sorted_results = collections.OrderedDict(sorted(ranking.items(), key=lambda val: val[1], reverse=True))
        top_k_items = {item: sorted_results[item] for item in list(sorted_results)[:k+1]}

        plot_images = {}
        for image_id in top_k_items.keys():
            if image_id!=test_image:
                image_path = os.path.join(test_folder_path, image_id)
                plot_images[image_path] = top_k_items[image_id]

        misc.plot_similar_images(plot_images)


