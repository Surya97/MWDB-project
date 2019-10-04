import sys
sys.path.insert(1, '../Phase1')

from similar_images import Similarity
from Decomposition import Decomposition
from Metadata import Metadata
import os
from pathlib import Path
import misc


task = input("Please specify the task number: ")
test_dataset_path = input("Please specify test folder path: ")
model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
decomposition = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")

if task == '1':
    k = int(input("Enter the number of latent features to consider: "))
    decomposition = Decomposition(decomposition, 20, model, test_dataset_path)
    decomposition.dimensionality_reduction()
    decomposition.decomposition_model.print_term_weight_pairs(k)

elif task == '2':
    image_id = input("Please specify the test image file name: ")
    m = int(input("Please specify the value of m: "))
    decomposition = Decomposition(decomposition, m, model, test_dataset_path)
    similarity = Similarity(model, image_id, m)
    similarity.get_similar_images(test_dataset_path, decomposition, reduced_dimension=True)

elif task == '3':
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
    decomposition = Decomposition(decomposition, k, model, test_dataset_path, metadata_images_list=metadata_images_list,
                                  metadata_label=metadata_label)
    decomposition.dimensionality_reduction()

else:
    print('Please enter the correct task number !')
