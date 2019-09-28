from features_images import FeaturesImages
import os
from pathlib import Path
import misc
from tqdm import tqdm
import collections

'''
Similarity measure base class. 'get_similar_images' function plots the top k images based on the 
similarity measure suitable for the model.
'''


class Similarity:
    def __init__(self, model_name, test_image_id, k):
        self.model_name = model_name
        self.test_image_id = test_image_id
        self.k = k

    def get_similar_images(self, test_folder):
        features_images = FeaturesImages(self.model_name)
        test_folder_path = os.path.join(Path(os.path.dirname(__file__)).parent, test_folder)
        test_image_path = os.path.join(test_folder_path, self.test_image_id)
        test_image_features = features_images.compute_image_features(test_image_path)
        model = features_images.get_model()
        dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), self.model_name)
        ranking = {}
        for image_id, feature_vector in tqdm(dataset_images_features.items()):
            distance = model.similarity_fn(test_image_features, feature_vector)
            ranking[image_id] = distance

        sorted_results = collections.OrderedDict(sorted(ranking.items(), key=lambda val: val[1],
                                                        reverse=model.reverse_sort))
        top_k_items = {item: sorted_results[item] for item in list(sorted_results)[:self.k + 1]}

        plot_images = {}
        for image_id in top_k_items.keys():
            if image_id != self.test_image_id:
                image_path = os.path.join(test_folder_path, image_id)
                plot_images[image_path] = top_k_items[image_id]
        print('test2')
        misc.plot_similar_images(plot_images)
