import sys
sys.path.insert(1, '../Phase1')
import misc
from KMeans import KMeans
from label_features import LabelFeatures
from Metadata import Metadata
from customlshash import MyCustomLSH
from DecisionTree import DecisionTree
from Feedback import Feedback
from visualize_clusters import VisualizeClusters
import pickle
import numpy as np

task = input("Please specify the task number: ")


if task == '2':
    c = int(input("Enter the number Of clusters:"))
    labelled_dataset_path = input('Enter labelled dataset path: ')
    unlabelled_dataset_path = input('Enter unlabelled dataset path: ')
    kmeans = KMeans(c)
    label_features = LabelFeatures(labelled_dataset_path=labelled_dataset_path,
                                   unlabelled_dataset_path=unlabelled_dataset_path)
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
    dorsal_cluster_visualization = VisualizeClusters(dorsal_features, dorsal_image_cluster_map)
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
    palmar_cluster_visualization = VisualizeClusters(palmar_features, palmar_image_cluster_map)
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

elif task == '4':
    classifier = input("1.SVM\n2.DT\n3.PPR\nSelect Classifier: ")
    labelled_dataset_path = input('Enter labelled dataset path: ')
    unlabelled_dataset_path = input('Enter unlabelled dataset path: ')

    label_features = LabelFeatures(labelled_dataset_path=labelled_dataset_path,
                                   unlabelled_dataset_path=unlabelled_dataset_path)
    label_features.set_features()
    dorsal_features = label_features.get_label_features('dorsal')
    palmar_features = label_features.get_label_features('palmar')
    unlabelled_features = label_features.get_unlabelled_images_decomposed_features()
    result = {}
    if classifier == 'DT':
        decisiontree = DecisionTree()
        decisiontree.generate_input_data(dorsal_features, palmar_features)
        dt = decisiontree.build_tree(decisiontree.dataset, 10, 1)

        for image_id, feature in unlabelled_features.items():
            feature = list(feature)
            feature.append(None)
            val = decisiontree.predict(dt, feature)
            if val == 0:
                result[image_id]='dorsal'
            elif val == 1:
                result[image_id]='palmar'

    #ACCURACY
    metadata = Metadata(metadatapath='Data/HandInfo.csv')
    images_dop_dict = metadata.getimagesdop_dict()
    print('Accuracy:', misc.getAccuracy(result, images_dop_dict))


elif task == '5':
    num_layers = int(input("Enter the number Of Layers:"))
    num_hashfunctions = int(input("Enter the number Of Hashes per layer:"))
    q_image_id = input("Enter The ImageId:")
    t = int(input('Enter the Value of t:'))
    lsh = MyCustomLSH(number_of_hashes_per_layer =num_hashfunctions, number_of_features =256, num_layers=num_layers)
    final_path = '../Phase2/pickle_files/LBP_PCA_11k.pkl'
    print('loading from pickle file path', final_path)
    infile = open(final_path, 'rb')
    dataset_features = pickle.load(infile)
    metadata = Metadata(metadatapath='Data/HandInfo.csv')
    images_dop_dict = metadata.getimagesdop_dict()

    for image_id, feature in dataset_features.items():
        lsh.add_to_index_structure(input_feature =feature, image_id=image_id)

    ret_val = lsh.query(dataset_features[q_image_id], num_results=t)
    print('Query Image:', q_image_id, images_dop_dict[q_image_id])
    result = {}
    for val in ret_val:
        result[val[1]] = val[2]
        print(val[1], images_dop_dict[val[1]], val[2])

    lsh.save_result(result)

elif task == '6' :
    r = int(input('Number Of Images you would like to label as Relevant:'))
    ir = int(input('Number of Images you would like to label as Irrelevant:'))
    final_path = '../Phase2/pickle_files/LBP_PCA_11k.pkl'
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
        rorir_map[num_image[ind]]=1
    while ir > 0:
        ind = int(input('Enter the Image Number to Label as Irrelevant:'))
        ir -= 1
        rorir_map[num_image[ind]] = 0

    feedback.generate_input_data(rorir_map, dataset_features)
    classifier = input("1.SVM\n2.DT\n3.PPR\n4.Prob\nSelect Classifier: ")

    if classifier == 'DT':
        decisiontree = DecisionTree()
        decisiontree.dataset = feedback.dataset
        dt = decisiontree.build_tree(decisiontree.dataset, 10, 1)
        for image_id, label in rorir_map.items():
            feature = dataset_features[image_id]
            feature = list(feature)
            feature.append(None)
            val = decisiontree.predict(dt, feature)
            rorir_map[image_id] = val

    for image_id, val in rorir_map.items():
        print(image_id, val)

