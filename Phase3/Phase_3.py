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
from SVM import SVM
from page_rank_util import PageRankUtil
import helper_functions
from tqdm import tqdm
from decision_tree import DecisionTreeClassifier
import random

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

    metadata = Metadata(metadatapath='Data/HandInfo.csv')
    result = {}

    if classifier != 'PPR':
        label_features = LabelFeatures(labelled_dataset_path=labelled_dataset_path,
                                       unlabelled_dataset_path=unlabelled_dataset_path, feature_name='HOG',
                                       decomposition_name='SVD')
        label_features.set_features()
        label_folder_features = helper_functions.get_main_features('LBP', labelled_dataset_path)

        dorsal_features = label_features.get_label_features('dorsal')
        palmar_features = label_features.get_label_features('palmar')
        # unlabelled_features = label_features.get_unlabelled_images_decomposed_features()
        unlabelled_features = helper_functions.get_main_features('LBP', unlabelled_dataset_path)
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
            svm = SVM()
            svm.generate_input_data(dorsal_features, palmar_features)
            svm.fit(svm.dataset)

            for image_id, feature in unlabelled_features.items():
                feature = list(feature)
                val = svm.predict(feature)
                if val.any() == 0:
                    result[image_id] = 'dorsal'
                elif val.any() == 1:
                    result[image_id] = 'palmar'
    else:
        unlabelled_images_list = list(helper_functions.get_images_list(unlabelled_dataset_path).keys())
        result = {}
        ppr = PageRankUtil(labelled_dataset_path, 30, 10, [])
        original_image_list = ppr.get_original_image_list()
        original_feature_map = ppr.get_original_image_feature_map()
        decomposition = ppr.get_decomposition()
        images_dop_map = metadata.getimagesdop_dict()
        # print(images_dop_map)
        for image in tqdm(unlabelled_images_list):
            ppr = PageRankUtil(labelled_dataset_path, 30, 10, [image], decomposition=decomposition,
                               unlabelled_image={image: unlabelled_dataset_path}, image_list=original_image_list,
                               feature_map=original_feature_map)
            ppr.page_rank_util()
            page_ranking = ppr.get_page_ranking()
            dorsal_count = 0
            palmar_count = 0
            # print('aspect of hand of ', image, 'is', metadata.get_label_value_image(image, 'aspectOfHand'))
            top_10_images = list(page_ranking.keys())[:10]
            # print(image, top_10_images)
            # print()
            for top_image in top_10_images:
                if 'dorsal' in images_dop_map[top_image]:
                    dorsal_count += 1
                else:
                    palmar_count += 1
            if dorsal_count > palmar_count:
                result[image] = 'dorsal'
            else:
                result[image] = 'palmar'

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

    ret_val, no_of_images = lsh.query(dataset_features[q_image_id], num_results=t)
    print('Query Image:', q_image_id, images_dop_dict[q_image_id])
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

    num_image = {}
    count = 1
    rorir_map = {}
    for image_id, val in task5_result.items():
        num_image[count] = image_id
        print(count, image_id)
        rorir_map[num_image[count]] = -1
        count += 1

    while r > 0:
        ind = int(input('Enter the Image Number to Label as Relevant:'))
        r -= 1
        rorir_map[num_image[ind]] = 1
    while ir > 0:
        ind = int(input('Enter the Image Number to Label as Irrelevant:'))
        ir -= 1
        rorir_map[num_image[ind]] = 0

    feedback.generate_input_data(rorir_map, dataset_features)
    classifier = input("1.SVM\n2.DT\n3.PPR\n4.Prob\nSelect Classifier: ")

    if classifier == 'DT':
        decisiontree = DecisionTreeClassifier()
        decisiontree.dataset = feedback.dataset
        dt = decisiontree.build_tree(decisiontree.dataset, 10, 1)
        for image_id, label in rorir_map.items():
            if rorir_map[image_id] == -1:
                feature = dataset_features[image_id]
                feature = list(feature)
                feature.append(None)
                val = decisiontree.predict(dt, feature)
                rorir_map[image_id] = val

    for image_id, val in rorir_map.items():
        print(image_id, val)

