from KMeans import KMeans
from label_features import LabelFeatures


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

    kmeans.fit(dorsal_features)
    similarity_val1 = kmeans.get_similarity_val(labelled_dataset_features=dorsal_features,
                                                unlabelled_dataset_features=unlabelled_features)

    kmeans.fit(palmar_features)
    similarity_val2 = kmeans.get_similarity_val(labelled_dataset_features=palmar_features,
                                                unlabelled_dataset_features=unlabelled_features)

elif task == '3':
    test_dataset_path = input("Enter test dataset path: ")




