from skimage import feature
import misc
import os
import collections
import numpy as np


def cosine_similarity(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    sum_xy = sum([(a-mean_x) * (b-mean_y) for a, b in zip(x, y)])
    sum_square_x = sum([(a-mean_x) * (a-mean_x) for a in x])
    sum_square_y = sum([(b-mean_y) * (b-mean_y) for b in y])
    cosine_sim = sum_xy / (pow(sum_square_x, 0.5) * pow(sum_square_y, 0.5))
    return cosine_sim


class Hog:
    def __init__(self, orientations, pixels_per_cell, cells_per_block):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def compute(self, image):
        return self.compute_hog(image)

    def compute_hog(self, image):
        (H, hogImage) = feature.hog(image, orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block, visualize=True)
        # misc.plot_image(hogImage)
        return H

    def get_similar_images(self, test_image_feature, k, test_folder_path, test_image):
        # misc.plot_image(misc.read_image(os.path.join(test_folder_path, test_image)))
        dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), 'HOG')
        cosine_similarity_ranking = {}
        for image_id, feature_vector in dataset_images_features.items():
            similarity = cosine_similarity(test_image_feature, feature_vector)
            cosine_similarity_ranking[image_id] = similarity

        sorted_hog_results = collections.OrderedDict(sorted(cosine_similarity_ranking.items(),
                                                            key=lambda val: val[1], reverse=True))
        top_k_items = {item: sorted_hog_results[item] for item in list(sorted_hog_results)[:k+1]}

        plot_images = {}
        for image_id in top_k_items.keys():
            if image_id!=test_image:
                image_path = os.path.join(test_folder_path, image_id)
                plot_images[image_path] = top_k_items[image_id]

        misc.plot_similar_images(plot_images)

