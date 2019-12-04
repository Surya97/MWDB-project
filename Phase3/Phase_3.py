import sys
sys.path.insert(1, '../Phase1')
import misc
from KMeans import KMeans
from label_features import LabelFeatures
from Metadata import Metadata
from customlshash import MyCustomLSH
from Feedback import Feedback
from visualize_clusters import VisualizeClusters
import pickle
import numpy as np
# from SVM import SVM
from svm_test import SVM, linear_kernel, polynomial_kernel, gaussian_kernel
from page_rank_util import PageRankUtil
import helper_functions
from tqdm import tqdm
from decision_tree import DecisionTreeClassifier
import random
import os
import collections
from ProbRF import *

task = input("Please specify the task number: ")


if task == '2':
    c = int(input("Enter the number Of clusters:"))
    labelled_dataset_path = input('Enter labelled dataset path: ')
    unlabelled_dataset_path = input('Enter unlabelled dataset path: ')
    kmeans = KMeans(c)
    label_features = LabelFeatures(labelled_dataset_path=labelled_dataset_path,
                                   unlabelled_dataset_path=unlabelled_dataset_path, feature_name='SIFT',
                                   decomposition_name='')
    label_features.set_features()
    dorsal_features = label_features.get_label_features('dorsal')
    palmar_features = label_features.get_label_features('palmar')
    unlabelled_features = label_features.get_unlabelled_images_decomposed_features()
    print('Computing clusters associated with dorsal-hand images...')
    temp_dictionary = list(dorsal_features.items())
    np.random.seed(23)
    np.random.shuffle(temp_dictionary)
    dorsal_features = dict(temp_dictionary)
    kmeans.fit(dorsal_features)

    # Visualizing dorsal image clusters
    dorsal_image_cluster_map = kmeans.get_image_cluster_map()
    dorsal_cluster_visualization = VisualizeClusters(dorsal_features, dorsal_image_cluster_map, 'dorsal')
    dorsal_cluster_visualization.plot()

    similarity_val1 = kmeans.get_similarity_val(labelled_dataset_features=dorsal_features,
                                                unlabelled_dataset_features=unlabelled_features)

    print('Computing clusters associated with palmar-hand images...')
    temp_dictionary = list(palmar_features.items())
    np.random.shuffle(temp_dictionary)
    palmar_features = dict(temp_dictionary)
    kmeans.fit(palmar_features)

    # Visualizing palmar image clusters
    palmar_image_cluster_map = kmeans.get_image_cluster_map()
    palmar_cluster_visualization = VisualizeClusters(palmar_features, palmar_image_cluster_map, 'palmar')
    palmar_cluster_visualization.plot()

    similarity_val2 = kmeans.get_similarity_val(labelled_dataset_features=palmar_features,
                                                unlabelled_dataset_features=unlabelled_features)
    result = {}
    for image_id in list(unlabelled_features.keys()):
        if similarity_val1[image_id] <= similarity_val2[image_id]:
            result[image_id] = 'dorsal'
        else:
            result[image_id] = 'palmar'

    print(result)

    #ACCURACY
    metadata = Metadata(metadatapath='Data/HandInfo.csv')
    images_dop_dict = metadata.getimagesdop_dict()
    print('Accuracy:', misc.getAccuracy(result, images_dop_dict))

elif task == '3':
    folder_path = input("Enter folder path: ")
    start_images = list(map(str, input("Enter 3 imageids: ").split()))
    k = int(input("Enter number of outgoing edges: "))
    m = int(input("Enter number of dominant images to show: "))
    pagerank = PageRankUtil(folder_path, k, m, start_images)
    pagerank.page_rank_util()
    pagerank.plot_k_similar()


elif task == '4':
    classifier = input("1.SVM\n2.DT\n3.PPR\nSelect Classifier: ")
    labelled_dataset_path = input('Enter labelled dataset path: ')
    unlabelled_dataset_path = input('Enter unlabelled dataset path: ')

    label_feature_name = 'LBP'
    if classifier == 'DT':
        label_feature_name = 'LBP'
    elif classifier == 'PPR':
        label_feature_name = 'HOG'

    result = {}
    print('Getting Labeled Image features from Phase 1')
    label_folder_features = helper_functions.get_main_features(label_feature_name, labelled_dataset_path)
    metadata = Metadata(list(label_folder_features.keys()), metadatapath='Data/HandInfo.csv')
    dorsal_images_list = metadata.get_specific_metadata_images_list({'aspectOfHand': 'dorsal'})
    palmar_images_list = metadata.get_specific_metadata_images_list({'aspectOfHand': 'palmar'})

    print('Getting unlabelled image features from Phase 1')
    unlabelled_features = helper_functions.get_main_features(label_feature_name, unlabelled_dataset_path)

    dorsal_features = {}
    palmar_features = {}

    for image in dorsal_images_list:
        dorsal_features[image] = label_folder_features[image]
    for image in palmar_images_list:
        palmar_features[image] = label_folder_features[image]

    if classifier == 'DT':
        decisiontree = DecisionTreeClassifier(max_depth=100)
        dorsal_images = list(dorsal_features.keys())
        palmar_images = list(palmar_features.keys())
        image_list = dorsal_images
        image_list.extend(palmar_images)
        random.shuffle(image_list)
        X = []
        y = [0]*len(image_list)

        for i in range(0, len(image_list)):
            image = image_list[i]
            if image in dorsal_features:
                y[i] = 0
            else:
                y[i] = 1
            X.append(label_folder_features[image])
        X = np.array(X)
        y = np.array(y)
        decisiontree.fit(X, y)
        for image_id, feature in unlabelled_features.items():
            val = decisiontree.predict([feature])
            print(image_id, val)
            if val[0] == 0:
                result[image_id] = 'dorsal'
            else:
                result[image_id] = 'palmar'

    elif classifier == 'SVM':
        # svm = SVM()
        # svm.generate_input_data(dorsal_features, palmar_features)
        # svm.fit(svm.dataset)
        # for image_id, feature in unlabelled_features.items():
        #     feature = list(feature)
        #     val = svm.predict(feature)
        #     print(image_id, val)
        #     if val.any() == 0:
        #         result[image_id] = 'dorsal'
        #     elif val.any() == 1:
        #         result[image_id] = 'palmar'
        dorsal_images = list(dorsal_features.keys())
        palmar_images = list(palmar_features.keys())
        image_list = dorsal_images
        image_list.extend(palmar_images)
        random.shuffle(image_list)
        X = []
        y = [0] * len(image_list)

        for i in range(0, len(image_list)):
            image = image_list[i]
            if image in dorsal_features:
                y[i] = -1
            else:
                y[i] = 1
            X.append(label_folder_features[image])
        X = np.array(X)
        y = np.array(y)
        svm = SVM()
        svm.fit(X, y)
        for image_id, feature in unlabelled_features.items():
            val = svm.predict([feature])
            print(image_id, val)
            if val[0] == 0:
                result[image_id] = 'dorsal'
            else:
                result[image_id] = 'palmar'

    elif classifier == 'PPR':
        unlabelled_images_list = list(unlabelled_features.keys())
        result = {}
        ppr = PageRankUtil(labelled_dataset_path, 10, 20, [], feature_name=label_feature_name)
        original_image_list = ppr.get_original_image_list()
        original_feature_map = ppr.get_original_image_feature_map()
        images_dop_map = metadata.getimagesdop_dict()
        # print(images_dop_map)
        for image in tqdm(unlabelled_images_list):
            ppr.set_unlabelled_image({image: unlabelled_dataset_path})
            ppr.set_start_images_list(image)
            ppr.set_image_list_and_feature_map(original_image_list, original_feature_map)
            ppr.initialize()
            ppr.page_rank_util()
            page_ranking = ppr.get_page_ranking()
            dorsal_count = 0
            palmar_count = 0
            top_10_images = list(page_ranking.keys())[:20]
            for top_image in top_10_images:
                if 'dorsal' in images_dop_map[top_image]:
                    dorsal_count += 1
                else:
                    palmar_count += 1
            if dorsal_count > palmar_count:
                result[image] = 'dorsal'
            else:
                result[image] = 'palmar'

    print(result)
    #ACCURACY
    images_dop_dict = metadata.getimagesdop_dict()
    print('Accuracy:', misc.getAccuracy(result, images_dop_dict))


elif task == '5':
    num_layers = int(input("Enter the number Of Layers:"))
    num_hashfunctions = int(input("Enter the number Of Hashes per layer:"))
    q_image_id = input("Enter The ImageId:")
    t = int(input('Enter the Value of t:'))
    lsh = MyCustomLSH(number_of_hashes_per_layer=num_hashfunctions, number_of_features=256, num_layers=num_layers)
    final_path = '../Phase2/pickle_files/HOG_SVD_11k.pkl'
    print('loading from pickle file path', final_path)
    infile = open(final_path, 'rb')
    dataset_features = pickle.load(infile)
    metadata = Metadata(metadatapath='Data/HandInfo.csv')
    images_dop_dict = metadata.getimagesdop_dict()

    for image_id, feature in dataset_features.items():
        lsh.add_to_index_structure(input_feature =feature, image_id=image_id)

    ret_val, no_of_images, unique_images = lsh.query(dataset_features[q_image_id], num_results=t)
    print('Query Image:', q_image_id, images_dop_dict[q_image_id])
    print("Total Number of Images:", no_of_images)
    print("Unique Images:", unique_images)
    result = {}
    for val in ret_val:
        result[val[0]] = val[1]
    misc.plot_similar_images(result)
    lsh.save_result(result)

elif task == '6':
    r = int(input('Number Of Images you would like to label as Relevant:'))
    ir = int(input('Number of Images you would like to label as Irrelevant:'))
    final_path = '../Phase2/pickle_files/HOG_SVD_11k.pkl'
    print('loading from pickle file path', final_path)
    infile = open(final_path, 'rb')
    dataset_features = pickle.load(infile)
    feedback = Feedback()
    task5_result = feedback.task5_result
    base_id=None
    num_image = {}
    count = 1
    rorir_map = {}
    for image_id, val in task5_result.items():
        image_id = os.path.basename(image_id)
        num_image[count] = image_id
        print(count, image_id)
        rorir_map[num_image[count]] = -1
        count += 1
    count=0
    while r > 0:
        ind = int(input('Enter the Image Number to Label as Relevant:'))
        r -= 1
        if count==0:
            base_id=num_image[ind]
            count += 1
        rorir_map[num_image[ind]] = 1
    while ir > 0:
        ind = int(input('Enter the Image Number to Label as Irrelevant:'))
        ir -= 1
        rorir_map[num_image[ind]] = 0

    feedback.generate_input_data(rorir_map, dataset_features)
    classifier = input("1.SVM\n2.DT\n3.PPR\n4.Prob\nSelect Classifier: ")

    if classifier == 'DT':
        decisiontree = DecisionTreeClassifier(max_depth=100)
        decisiontree.fit(feedback.X, feedback.y)
        for image_id, label in rorir_map.items():
            if rorir_map[image_id] == -1:
                feature = dataset_features[image_id]
                val = decisiontree.predict([feature])
                rorir_map[image_id] = val[0]
    elif classifier == 'Prob':
        feedback.generate_input_data_set(rorir_map, dataset_features)
        dataset = feedback.dataset
        model = division_by_class(dataset)
        # define a new record
        for image_id, label in rorir_map.items():
            if rorir_map[image_id] == -1:
                feature = dataset_features[image_id]
                val = predict(model, feature)
                rorir_map[image_id] = int(val)
                
    old_list_images=list()
    for image_id, val in rorir_map.items():
        old_list_images.append(image_id)


    new_ordered_images = [(image_id,
                   feedback.euclidean_distance(dataset_features[base_id], dataset_features[image_id])) for image_id in old_list_images]
    new_ordered_images.sort(key=lambda v: v[1])

    result = collections.OrderedDict()
    for val in new_ordered_images:
        result[val[0]] = val[1]
    final_result = collections.OrderedDict()
    for val, dist_val in result.items():
        if rorir_map[val] == 1:
            print(val, rorir_map[val])
            final_result[val]=dist_val

    for val, dist_val in result.items():
        if rorir_map[val] == 0:
            print(val, rorir_map[val])
            final_result[val]=dist_val

    feedback.save_result(final_result)