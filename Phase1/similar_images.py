from features_images import FeaturesImages
import os
from pathlib import Path


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
        model.get_similar_images(test_image_features, self.k, test_folder_path, self.test_image_id)
