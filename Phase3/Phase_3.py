# Phase 3 CLI
import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(2, '../Phase2')
import os
from pathlib import Path
import misc
from similar_images import Similarity
from Decomposition import Decomposition

task = input("Please specify the task number: ")

if task == '3':
    test_dataset_path = 'Data/testdata'
    reduced_images_features = {}

    reduced_dimension_pickle_path = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                 'Phase2', 'pickle_files')
    print("path:", reduced_dimension_pickle_path)
    print(os.path.exists(os.path.join(reduced_dimension_pickle_path,
                                        'LBP'+'_'+'PCA'+'.pkl')))
    if not (os.path.exists(os.path.join(reduced_dimension_pickle_path,
                                        'LBP'+'_'+'PCA'+'.pkl'))):
        print('Pickle file not found for the Particular (model,Reduction)')
        print('Runnning Task1 for the Particular (model,Reduction) to get the pickle file')
        #decomposition = Decomposition('PCA', 20, 'LBP', 'Data/testdata')
        #decomposition.dimensionality_reduction()
        #reduced_images_features = misc.load_from_pickle(reduced_dimension_pickle_path, 'LBP_PCA', 20)

    else :
        print('Getting the Decomposed Features from Pickle File:')
        reduced_images_features = misc.load_from_pickle(reduced_dimension_pickle_path, 'LBP_PCA', 20)

    k = int(input("Enter the number values of k to create the graph: "))
    decomposition = Decomposition('PCA', 20, 'LBP',test_dataset_path )
    similarity = Similarity('LBP', '', k)

    k_images = []
    m1 = {}
    m2 = {}
    m3 = {}
    count = 0
    for image_id in reduced_images_features.keys():
        m1[image_id] = count
        m2[count] = image_id
        similarity.set_test_image_id(image_id)
        k_images = similarity.get_similar_image_ids(test_dataset_path, decomposition, reduced_dimension=True)
        print(k_images)
        #add the edges in the graph here
        count = count +1


    g = {"a": ["d"],
         "b": ["c"],
         "c": ["b", "c", "d", "e"],
         "d": ["a", "c"],
         "e": ["c"],
         "f": []
         }


