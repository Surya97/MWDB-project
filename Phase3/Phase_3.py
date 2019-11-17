# Phase 3 CLI

from KMeans import KMeans

task = input("Please specify the task number: ")

if task == '2':
    c = int(input("Enter the number Of clusters:"))
    labelled_dataset_path = input('Enter labelled dataset path: ')
    unlabelled_dataset_path = input('Enter unlabelled dataset path: ')
    x = KMeans(5, labelled_dataset_path=labelled_dataset_path, unlabelled_dataset_path=unlabelled_dataset_path )
    x.set_label_features()
    x.fit(x.get_label_features('dorsal'))
    similarity_val1 = x.get_similarity_val(labelled_dataset_path)
    x.fit(x.get_label_features('palmar'))
    similarity_val2 = x.get_similarity_val(unlabelled_dataset_path)

elif task == '3':
    test_dataset_path = 'Data/Dataset2'




