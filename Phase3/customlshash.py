import os
import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
from pathlib import Path
import misc
import os
import numpy as np
wind_size = 5
offset = np.random.randint(wind_size)


def euclidean_dist_square(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)


class MyCustomLSH(object):

    def __init__(self, number_of_hashes_per_layer, number_of_features, num_layers=2):
        self.num_layers = num_layers
        self.number_of_hashes_per_layer = number_of_hashes_per_layer
        self.number_of_features = number_of_features
        self.random_planes = [np.random.randn(self.number_of_hashes_per_layer, self.number_of_features)
                              for _ in range(self.num_layers)]
        self.layers = [dict() for i in range(self.num_layers)]

    def get_combined_hash_value(self, planes, input_point, j):
        input_point = np.array(input_point)
        projections = planes.dot(input_point)
        # projections = projections/50
        # val= "".join([str(int(i)+j) for i in projections])
        # return val
        return "".join(['1' if i > 0 else '0' for i in projections])

    def add_to_index_structure(self, input_feature, image_id=''):
        value = tuple(input_feature)
        for i, layer in enumerate(self.layers):
            layer.setdefault(self.get_combined_hash_value(self.random_planes[i], input_feature, 0), []).append((value, image_id))

    def query(self, feature, num_results=None, distance_func=None):
        image_hits = set()
        calculate_distance = euclidean_dist_square

        for i, layer in enumerate(self.layers):
            combined_hash_value = self.get_combined_hash_value(self.random_planes[i], feature, 0)
            image_hits.update(layer.get(combined_hash_value, []))
        j=1

        while len(image_hits)<num_results:
            for i, layer in enumerate(self.layers):
                combined_hash_value = self.get_combined_hash_value(self.random_planes[i], feature, j)
                image_hits.update(layer.get(combined_hash_value, []))
                combined_hash_value = self.get_combined_hash_value(self.random_planes[i], feature, 0-j)
                image_hits.update(layer.get(combined_hash_value, []))
                # print(len(image_hits))
            j+=1
        # image_hits = [(hit_tuple[0], hit_tuple[1], calculate_distance(feature, np.asarray(hit_tuple[0])))
        #               for hit_tuple in image_hits]
        image_hits = [(os.path.join('../data/Hands', hit_tuple[1]),
                       calculate_distance(feature, hit_tuple[0])) for hit_tuple in image_hits]
        image_hits.sort(key=lambda v: v[1])
        
        result = image_hits[:num_results] if num_results else image_hits

        return result, len(image_hits), len(set(image_hits))

    def save_result(self, result):
        reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        misc.save2pickle(result, reduced_pickle_file_folder, 'Task_5_Result')