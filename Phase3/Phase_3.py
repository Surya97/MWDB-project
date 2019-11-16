# Phase 3 CLI

from KMeans import KMeans
task = input("Please specify the task number: ")
import os
from pathlib import Path
import misc


if task == '2':
    c = int(input("Enter the Number Of Clusters:"))
    labelled_dataset_path = 'Data/testdata'
    unlabelled_dataset_path = 'Data/testdata'
    x = KMeans(5, labeled_dataset_path=labelled_dataset_path, unlabeled_dataset_path=unlabelled_dataset_path )
    x.set_label_features()
    x.fit(x.get_label_features('dorsal'))
    similarity_val1 = x.get_similarity_val(unlabelled_dataset_path)
    #x.fit(x.get_label_features('palmar'))
    #similarity_val2 = x.get_similarity_val(unlabelled_dataset_path)



elif task == '3':
    test_dataset_path = 'Data/testdata'




