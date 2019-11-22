
import os
import numpy as np
wind_size = 5
offset = np.random.randint(wind_size)


def euclidean_dist(x, y):
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

def euclidean_dist_square(x, y):
        diff = np.array(x) - y
        return np.dot(diff, diff)

def euclidean_dist_centred(x, y):
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

class MyCustomLSH(object):

    def __init__(self, number_of_hashes_per_layer, number_of_features, num_layers=2):
        self.num_layers = num_layers
        self.number_of_hashes_per_layer = number_of_hashes_per_layer
        self.number_of_features = number_of_features
        self.random_planes = [np.random.randn(self.number_of_hashes_per_layer, self.number_of_features)
                              for _ in range(self.num_layers)]
        self.layers = [dict() for i in range(self.num_layers)]

    def get_combined_hash_value(self, planes, input_point):
        input_point = np.array(input_point)
        projections = np.dot(planes, input_point)
        return "".join(['1' if i > 0 else '0' for i in projections])

    def add_to_index_structure(self, input_feature, image_id=''):
        value = tuple(input_feature)
        for i, layer in enumerate(self.layers):
            layer.setdefault(self.get_combined_hash_value(self.random_planes[i], input_feature), []).append((value, image_id))

    def query(self, feature, num_results=None, distance_func=None, image_id=''):
        candidates = set()
        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "euclidean":
            d_func = euclidean_dist_square
        elif distance_func == "true_euclidean":
            d_func = euclidean_dist
        else:
            raise ValueError("The distance function name is invalid.")

        for i, layer in enumerate(self.layers):
            binary_hash = self.get_combined_hash_value(self.random_planes[i], feature)
            candidates.update(layer.get(binary_hash, []))

        candidates = [(cand[0], cand[1], d_func(feature, np.asarray(cand[0])))
                      for cand in candidates]
        candidates.sort(key=lambda v: v[2])
        print('Query Image:', image_id)
        return candidates[:num_results] if num_results else candidates

