import sys
sys.path.insert(1, '../Phase1')
import misc
from KMeans import KMeans
from label_features import LabelFeatures
from Metadata import Metadata
from customlshash import MyCustomLSH
from DecisionTree import DecisionTree
import pickle

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
    kmeans.fit(dorsal_features)
    similarity_val1 = kmeans.get_similarity_val(labelled_dataset_features=dorsal_features,
                                                unlabelled_dataset_features=unlabelled_features)
    print('Computing clusters associated with palmar-hand images...')
    kmeans.fit(palmar_features)
    similarity_val2 = kmeans.get_similarity_val(labelled_dataset_features=palmar_features,
                                                unlabelled_dataset_features=unlabelled_features)

    result={}
    for image_id in list(unlabelled_features.keys()):
        if similarity_val1[image_id] <= similarity_val2[image_id]:
            result[image_id]='dorsal'
        else:
            result[image_id]='palmar'

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
        decisiontree.build_tree(decisiontree.dataset, 10, 1)


        for image_id, feature in unlabelled_features.items():
            feature = list(feature)
            feature.append(None)
            val = decisiontree.predict(decisiontree, feature)
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
    image_id = input("Enter The ImageId:")
    lsh = MyCustomLSH(number_of_hashes_per_layer =6, number_of_features =256, num_layers=5)
    final_path = '../Phase2/pickle_files/LBP_PCA_11k.pkl'
    print('loading from pickle file path', final_path)
    infile = open(final_path, 'rb')
    dataset_features = pickle.load(infile)
    metadata = Metadata(metadatapath='Data/HandInfo.csv')
    images_dop_dict = metadata.getimagesdop_dict()

    for image_id, feature in dataset_features.items():
        lsh.add_to_index_structure(input_feature =feature, image_id=image_id)

    ret_val = lsh.query(dataset_features[image_id], image_id=image_id, num_results=20)
    print('Query Image:', image_id, images_dop_dict[image_id])
    for val in ret_val:
        print(val[1], images_dop_dict[val[1]])