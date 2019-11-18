import sys
sys.path.insert(1, '../Phase1')

from similar_images import Similarity
from Decomposition import Decomposition
from Metadata import Metadata
import os
from pathlib import Path
import misc
from NMF import NMFModel
import numpy as np
import pandas as pd

task = input("Please specify the task number: ")
test_dataset_path = input("Please specify test folder path: ")

if task == '1':
    model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
    decomposition_model = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")
    k = int(input("Enter the number of latent features to consider: "))
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_path)
    decomposition.dimensionality_reduction()
    decomposition.decomposition_model.print_term_weight_pairs(k)

elif task == '2':
    model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
    decomposition_model = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")
    image_id = input("Please specify the test image file name: ")
    k = int(input("Please specify the number of components : "))
    m = int(input("Please specify the value of m: "))
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_path)
    similarity = Similarity(model, image_id, m)
    similarity.get_similar_images(test_dataset_path, decomposition, reduced_dimension=True)

elif task == '3':
    model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
    decomposition_model = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")
    test_dataset_folder_path = os.path.abspath(
        os.path.join(Path(os.getcwd()).parent, test_dataset_path))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    metadata = Metadata(images_list)
    label = int(input("1.Left-Hand\n2.Right-Hand\n3.Dorsal\n4.Palmar\n"
                      "5.With accessories\n6.Without accessories\n7.Male\n8.Female\n"
                      "Please choose an option: "))
    label_interpret_dict = {
        1: {"aspectOfHand": "left"},
        2: {"aspectOfHand": "right"},
        3: {"aspectOfHand": "dorsal"},
        4: {"aspectOfHand": "palmar"},
        5: {"accessories": 1},
        6: {"accessories": 0},
        7: {"gender": "male"},
        8: {"gender": "female"}
    }

    metadata_images_list = metadata.get_specific_metadata_images_list(label_interpret_dict.get(label))
    k = int(input("Please specify the number of components : "))
    metadata_label = ''
    for key, value in label_interpret_dict.get(label).items():
        metadata_label = key + '_' + str(value)
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_path,
                                  metadata_images_list=metadata_images_list, metadata_label=metadata_label)
    decomposition.dimensionality_reduction()

elif task == '4':
    model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
    decomposition_model = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")
    test_dataset_folder_path = os.path.abspath(
        os.path.join(Path(os.getcwd()).parent, test_dataset_path))
    k = int(input("Please specify the number of components : "))
    test_image_id = input("Please specify test image ID: ")
    m = int(input("Please specify the value of m: "))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    metadata = Metadata(images_list)
    label = int(input("1.Left-Hand\n2.Right-Hand\n3.Dorsal\n4.Palmar\n"
                      "5.With accessories\n6.Without accessories\n7.Male\n8.Female\n"
                      "Please choose an option: "))
    label_interpret_dict = {
        1: {"aspectOfHand": "left"},
        2: {"aspectOfHand": "right"},
        3: {"aspectOfHand": "dorsal"},
        4: {"aspectOfHand": "palmar"},
        5: {"accessories": 1},
        6: {"accessories": 0},
        7: {"gender": "male"},
        8: {"gender": "female"}
    }

    metadata_images_list = metadata.get_specific_metadata_images_list(
        label_interpret_dict.get(label))
    metadata_images_list.append(test_image_id)
    metadata_label = ''
    for key, value in label_interpret_dict.get(label).items():
        metadata_label = key + '_' + str(value)
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_path,
                                  metadata_images_list=metadata_images_list, metadata_label=metadata_label)

    decomposition.dimensionality_reduction()
    pickle_file_path = model + "_" + decomposition_model + "_" + metadata_label
    similarity = Similarity(model, test_image_id, m+1)
    similarity.get_similar_images(test_dataset_path, decomposition=decomposition, reduced_dimension=True,
                                  metadata_pickle=pickle_file_path)

elif task == '5':
    model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
    decomposition_model = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")
    test_dataset_folder_path = os.path.abspath(
        os.path.join(Path(os.getcwd()).parent, test_dataset_path))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    metadata = Metadata(images_list)
    label = int(input("1.Left-Hand\n2.Right-Hand\n3.Dorsal\n4.Palmar\n"
                      "5.With accessories\n6.Without accessories\n7.Male\n8.Female\n"
                      "Please choose an option: "))
    label_interpret_dict = {
        1: {"aspectOfHand": "left"},
        2: {"aspectOfHand": "right"},
        3: {"aspectOfHand": "dorsal"},
        4: {"aspectOfHand": "palmar"},
        5: {"accessories": 1},
        6: {"accessories": 0},
        7: {"gender": "male"},
        8: {"gender": "female"}
    }

    metadata_images_list = metadata.get_specific_metadata_images_list(label_interpret_dict.get(label))
    metadata_label = ''
    for key, value in label_interpret_dict.get(label).items():
        metadata_label = key + '_' + str(value)

    k = int(input("Please specify the number of components : "))
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_path,
                                  metadata_images_list=metadata_images_list, metadata_label=metadata_label)

    decomposition.dimensionality_reduction()
    test_image_id = input("Please specify test image ID: ")
    pickle_file_path = model + "_" + decomposition_model + "_" + metadata_label

    labels_list = ['Left_Right', 'Dorsal_Palmar', 'Gender', 'Accessories']
    metadata_given = Metadata(metadata_images_list)
    metadata_given.set_unlabeled_image_features(model, test_image_id, decomposition)
    metadata_given.set_metadata_image_features(pickle_file_path)

    unlabeled_list = []
    for label in labels_list:
        unlabeled_list.append(metadata_given.get_binary_label(label))

    print('Labels of the unlabeled image are: ')
    print(unlabeled_list)

elif task == '6':
    test_dataset_folder_path = os.path.abspath(
        os.path.join(Path(os.getcwd()).parent, test_dataset_path))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    metadata = Metadata(images_list)
    subject_id = int(input("Please input the subject Id : "))
    sub_sub_list = metadata.sub_sub_list(subject_id)
    if sub_sub_list[0] == tuple([-1, -1]):
        print('Subject not present in the given dataset')
    else:
        print(sub_sub_list[0])
        print(sub_sub_list[1])
        print(sub_sub_list[2])
    metadata.plot_subjects(subject_id, sub_sub_list, test_dataset_folder_path)


elif task == '7':
    k = int(input("Enter the number of latent features to consider: "))
    test_dataset_folder_path = os.path.abspath(
        os.path.join(Path(os.getcwd()).parent, test_dataset_path))
    decomposition = Decomposition(feature_extraction_model_name='SIFT', test_folder_path=test_dataset_folder_path)
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    metadata = Metadata(images_list)
    sub_sub_matrix = metadata.subject_matrix()
    nmf = NMFModel(sub_sub_matrix, k, images_list)
    nmf.decompose()
    print('Decomposition Complete')
    nmf.print_term_weight_pairs(k)

elif task == '8':
    test_dataset_folder_path = os.path.abspath(
        os.path.join(Path(os.getcwd()).parent, test_dataset_path))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    metadata = Metadata(images_list)
    binary_image_metadata_matrix = metadata.get_binary_image_metadata()
    k = int(input("Enter the number of latent features to consider: "))
    nmf = NMFModel(binary_image_metadata_matrix, k, images_list)
    nmf.decompose()
    print('Decomposition Complete')
    nmf.print_term_weight_pairs(k)

elif task =='9':
    model = "LBP"
    decomposition_model = "PCA"
    test_dataset_pathTask4 = input("Please enter the labeled images path")
#    test_dataset_pathTask4 = 'D:\MWDBPhase31\phase3_sample_data\phase3_sample_data\Labelled\Set1'
    test_dataset_folder_path = os.path.abspath(
            os.path.join(Path(os.getcwd()).parent, test_dataset_pathTask4))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    path = input("Please enter the path of labeled images CSV")
#    path = 'D:\MWDBPhase31\phase3_sample_data\phase3_sample_data\labelled_set1.csv'
    metadata = Metadata(images_list, path)
    k = int(20)
    label_interpret_dict = {
        3: {"aspectOfHand": "dorsal"},
        4: {"aspectOfHand": "palmar"},
        5: {"aspectOfHand": "unlabeled"}
    }
    #lDorsal MeataData
    label = int(3)
    metadata_images_list = metadata.get_specific_metadata_images_list(label_interpret_dict.get(label))
    metadata_label = ''
    for key, value in label_interpret_dict.get(label).items():
        metadata_label = key + '_' + str(value)
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_pathTask4,
                                      metadata_images_list=metadata_images_list, metadata_label=metadata_label)
    decomposition.dimensionality_reduction()
    pickle_file_path = model + "_" + decomposition_model + "_" + metadata_label
    metadata_given_Dorsal = Metadata(metadata_images_list, path)
    #metadata_given.set_unlabeled_image_features(model, test_image_id, decomposition)
    metadata_given_Dorsal.set_metadata_image_features(pickle_file_path)
    
    #Palmer MetaData
    label = int(4)
    metadata_images_list = metadata.get_specific_metadata_images_list(label_interpret_dict.get(label))
    metadata_label = ''
    for key, value in label_interpret_dict.get(label).items():
        metadata_label = key + '_' + str(value)
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_pathTask4,
                                      metadata_images_list=metadata_images_list, metadata_label=metadata_label)
    decomposition.dimensionality_reduction()
    pickle_file_path = model + "_" + decomposition_model + "_" + metadata_label
    metadata_given_Palmer = Metadata(metadata_images_list, path)
    #metadata_given.set_unlabeled_image_features(model, test_image_id, decomposition)
    metadata_given_Palmer.set_metadata_image_features(pickle_file_path)
    
    #Unlabled MetaData
    label = int(5)
    test_dataset_pathTask4 = input("Please enter the un-labeled images path")
#    test_dataset_pathTask4 = "D:\\MWDBPhase31\\phase3_sample_data\\phase3_sample_data\\Unlabelled\\Set 1"
    test_dataset_folder_path = os.path.abspath(
            os.path.join(Path(os.getcwd()).parent, test_dataset_pathTask4))
    images_list = list(misc.get_images_in_directory(test_dataset_folder_path).keys())
    path = input("Please enter the path of unlabeled images CSV")
#    path ="D:\\MWDBPhase31\\phase3_sample_data\\phase3_sample_data\\unlabelled_set1.csv"

    metadata = Metadata(images_list, path)
    metadata_images_list = metadata.get_specific_metadata_images_list(label_interpret_dict.get(label), True)
    metadata_label = ''
    for key, value in label_interpret_dict.get(label).items():
        metadata_label = key + '_' + str(value)
    decomposition = Decomposition(decomposition_model, k, model, test_dataset_pathTask4,
                                      metadata_images_list=metadata_images_list, metadata_label=metadata_label)
    decomposition.dimensionality_reduction()
    #pickle_file_path = model + "_" + decomposition_model + "_" + metadata_label

    pickle_file_path = model + "_" + decomposition_model + "_" + metadata_label
    metadata_given_Unlabled = Metadata(metadata_images_list, path)
    #metadata_given.set_unlabeled_image_features(model, test_image_id, decomposition)
    metadata_given_Unlabled.set_metadata_image_features(pickle_file_path)

    labels_list = ['Dorsal', 'Palmar']

    X_train = []

    class_1_images = metadata_given_Dorsal.test_images_list
    class_0_images = metadata_given_Palmer.test_images_list

    class_1_images_features = []
    calss1X = pd.DataFrame();
    class_0_images_features = []
    for image in class_1_images:
        class_1_images_features.append(metadata_given_Dorsal.metadata_images_features[image])
    for image in class_0_images:
        class_0_images_features.append(metadata_given_Palmer.metadata_images_features[image])

else:
    print('Please enter the correct task number !')