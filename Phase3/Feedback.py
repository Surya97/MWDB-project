import os
import sys
from pathlib import Path
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
import misc
import numpy as np

class Feedback:
    def __init__(self):
        self.task5_result = None
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        self.set_task5_result()
        self.dataset = list()
        self.X = None
        self.y = None
        self.dataset=list()

    def generate_input_data_set(self, rorir_map, dataset_features):
        for image_id, label in rorir_map.items():
            image_id = os.path.basename(image_id)
            if label==0 or label==1:
                feat = dataset_features[image_id].tolist()
                feat+=[label]
                self.dataset.append(np.array(feat))
        return

    def set_task5_result(self):
        self.task5_result = misc.load_from_pickle(self.reduced_pickle_file_folder, 'Task_5_Result')

    def generate_input_data(self, rorir_map, dataset_features):
        X = []
        y = []

        for image_id, label in rorir_map.items():
            image_id = os.path.basename(image_id)
            if label == 0 or label == 1:
                X.append(dataset_features[image_id])
                y+=[rorir_map[image_id]]
        X = np.array(X)
        y = np.array(y)
        self.X=X
        self.y=y

        return

    def euclidean_distance(self, dist1, dist2):
        return (sum([(a - b) ** 2 for a, b in zip(dist1, dist2)])) ** 0.5

    def save_result(self, result):
        reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        misc.save2pickle(result, reduced_pickle_file_folder, 'Task_5_Result')