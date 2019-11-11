# Phase 3 CLI

from KMeans import KMeans
task = input("Please specify the task number: ")
import os
from pathlib import Path

if task == '3':
    test_dataset_path = 'Data/testdata'

    x = KMeans(5,test_dataset_path=test_dataset_path)
    x.set_label_features()


