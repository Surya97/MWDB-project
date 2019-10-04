import os
import numpy as np
import pandas as pd
from pathlib import Path


class Metadata:
    def __init__(self, test_images_list=None):
        self.test_images_list = test_images_list
        self.metadata_file_path = os.path.join(Path(os.path.dirname(__file__)).parent, 'data/HandInfo.csv')
        self.images_metadata = None
        self.set_images_metadata()

    def get_images_metadata(self):
        return self.images_metadata

    def set_images_metadata(self):
        self.images_metadata = pd.read_csv(self.metadata_file_path)

    def get_specific_metadata_images_list(self, feature_dict=None):

        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata

        # print('test_images_list', self.test_images_list)

        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        if feature_dict is not None:
            aspect_of_hand = ''
            if feature_dict.get('aspectOfHand'):
                aspect_of_hand += feature_dict.get('aspectOfHand')

            accessories = feature_dict.get('accessories')

            gender = feature_dict.get('gender')

            if aspect_of_hand != '':
                filtered_images_metadata = filtered_images_metadata[
                    (filtered_images_metadata['aspectOfHand'].str.contains(aspect_of_hand))]

            if accessories:
                filtered_images_metadata = filtered_images_metadata[
                    (filtered_images_metadata['accessories'] == accessories)]

            if gender:
                filtered_images_metadata = filtered_images_metadata[
                    (filtered_images_metadata['gender']) == gender]

        images_list = filtered_images_metadata['imageName'].tolist()
        return images_list

