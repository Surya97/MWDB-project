import os
import sys
from pathlib import Path
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
import misc


class Feedback:
    def __init__(self):
        self.task5_result = None
        self.reduced_pickle_file_folder = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                       'Phase2', 'pickle_files')
        self.set_task5_result()
        self.dataset = list()

    def set_task5_result(self):
        self.task5_result = misc.load_from_pickle(self.reduced_pickle_file_folder, 'Task_5_Result')

    def generate_input_data(self, rorir_map, dataset_features):
        for image_id, label in rorir_map.items():
            if label == 0 or label == 1:
                feat = list(dataset_features[image_id])
                feat.append(rorir_map[image_id])
                self.dataset.append(feat)
        return

