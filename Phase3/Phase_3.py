# Phase 3 CLI

from KMeans import KMeans
task = input("Please specify the task number: ")
import os
from pathlib import Path
import misc
from label_features import LabelFeatures

if task == '2':
    c = int(input("Enter the Number Of Clusters:"))
    labeled_dataset_path = 'Data/testdata'
    unlabeled_dataset_path = 'Data/testdata'
    labelfeatures=LabelFeatures(labeled_dataset_path=labeled_dataset_path,unlabeled_dataset_path=unlabeled_dataset_path )
    labelfeatures.set_features()
    #x = KMeans(5, labeled_dataset_path=labelled_dataset_path, unlabeled_dataset_path=unlabelled_dataset_path )
    print(len(labelfeatures.get_unlabeled_dataset_features().items()))
    #x.fit(x.get_label_features('palmar'))
    #similarity_val2 = x.get_similarity_val(unlabelled_dataset_path)



elif task == '3':
    test_dataset_path = 'Data/testdata'




