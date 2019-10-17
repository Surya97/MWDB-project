import os
import numpy as np
import pandas as pd
from pathlib import Path
from similar_images import Similarity
import misc
from prettytable import PrettyTable


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

            aspect_of_hand = feature_dict.get('aspectOfHand')

            accessories = feature_dict.get('accessories')

            gender = feature_dict.get('gender')

            if aspect_of_hand:
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

    def sub_sub_list(self, model, decomposition,sub1):
        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata
        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        subject_map = {}
        sub_ids_list = filtered_images_metadata['id'].unique().tolist()

        for sub_id in sub_ids_list:
            is_subject_id = filtered_images_metadata['id'] == sub_id
            subject_map[sub_id] = filtered_images_metadata[is_subject_id]


        reduced_dimension_pickle_path = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                     'Phase2', 'pickle_files')
        #for now taking - number of latent semantics as 20(max_val)
        dataset_images_features = misc.load_from_pickle(reduced_dimension_pickle_path,
                                                        model + '_' + decomposition, 20)
        similarity_list_of_pair = [0]
        for sub2 in sub_ids_list:
            sub_sub_val = self.subject_subject_similarity(subject_map[sub1], subject_map[sub2], model,
                                                          dataset_images_features)

            similarity_list_of_pair.append(tuple([sub_sub_val, sub2]))
        similarity_list_of_pair = similarity_list_of_pair[1:]
        similarity_list_of_pair = sorted(similarity_list_of_pair, key=lambda x: x[0])

        return similarity_list_of_pair


    def subject_matrix(self, model, decomposition):
        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata
        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        subject_map = {}
        sub_ids_list = filtered_images_metadata['id'].unique().tolist()
        sub_ids_list.sort() #just to look in sorted order OF SUBJECT IDS IN THE MATRIX
        #print(sub_ids_list)
        for sub_id in sub_ids_list:
            is_subject_id = filtered_images_metadata['id'] == sub_id
            subject_map[sub_id] = filtered_images_metadata[is_subject_id]


        reduced_dimension_pickle_path = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                     'Phase2', 'pickle_files')
        #for now taking - number of latent semantics as 20(max_val)
        dataset_images_features = misc.load_from_pickle(reduced_dimension_pickle_path,
                                                        model + '_' + decomposition, 20)
        similarity_matrix = []

        for sub1 in sub_ids_list:
            similarity_row = []
            similarity_row_pair = [0]
            for sub2 in sub_ids_list:
                if sub1 == sub2:
                    similarity_row = similarity_row + [-1]  #when comparing between two same subjects, i am ignoring
                else:
                    sub_sub_val = self.subject_subject_similarity(subject_map[sub1],subject_map[sub2], model, dataset_images_features)
                    similarity_row = similarity_row + sub_sub_val

            similarity_matrix.append(similarity_row)

        p = PrettyTable()
        p.add_row(['SUBJECT/SUBJECT'] + sub_ids_list)
        i=0
        for row in similarity_matrix:
            row = [sub_ids_list[i]] + row
            p.add_row(row)
            i = i+1
        print(p.get_string(header=False, border=False))

        return similarity_matrix


    def subject_subject_similarity(self,data_frame1,data_frame2, model, dataset_images_features):

        similarity_val = 0
        similarity = Similarity(model, '', 0)


        #dorsal left
        is_dorsal_left1 = data_frame1['aspectOfHand'] == 'dorsal left'
        list1 = data_frame1[is_dorsal_left1]['imageName'].tolist()
        is_dorsal_left2 = data_frame2['aspectOfHand'] == 'dorsal left'
        list2 = data_frame2[is_dorsal_left2]['imageName'].tolist()

        for image_id in list1:
            similarity.set_test_image_id(image_id)
            similarity_val = similarity_val + similarity.get_similarity_value(list2, dataset_images_features)


        #dorsal right
        is_dorsal_right1 = data_frame1['aspectOfHand'] == 'dorsal right'
        list1 = data_frame1[is_dorsal_right1]['imageName'].tolist()
        is_dorsal_right2 = data_frame2['aspectOfHand'] == 'dorsal right'
        list2 = data_frame2[is_dorsal_right2]['imageName'].tolist()

        for image_id in list1:
            similarity.set_test_image_id(image_id)
            similarity_val = similarity_val + similarity.get_similarity_value(list2, dataset_images_features)


        #palmar left
        is_palmar_left1 = data_frame1['aspectOfHand'] == 'palmar left'
        list1 = data_frame1[is_palmar_left1]['imageName'].tolist()
        is_palmar_left2 = data_frame2['aspectOfHand'] == 'palmar left'
        list2 = data_frame2[is_palmar_left2]['imageName'].tolist()

        for image_id in list1:
            similarity.set_test_image_id(image_id)
            similarity_val = similarity_val + similarity.get_similarity_value(list2, dataset_images_features)


        #palmar right
        is_palmar_right1 = data_frame1['aspectOfHand'] == 'palmar right'
        list1 = data_frame1[is_palmar_right1]['imageName'].tolist()
        is_palmar_right2 = data_frame2['aspectOfHand'] == 'palmar right'
        list2 = data_frame2[is_palmar_right2]['imageName'].tolist()

        for image_id in list1:
            similarity.set_test_image_id(image_id)
            similarity_val = similarity_val + similarity.get_similarity_value(list2, dataset_images_features)


        return [similarity_val]
