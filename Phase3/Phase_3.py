from KMeans import KMeans
from label_features import LabelFeatures
from Metadata import Metadata

task = input("Please specify the task number: ")


if task == '2':
    c = int(input("Enter the number Of clusters:"))
    labelled_dataset_path = input('Enter labelled dataset path: ')
    unlabelled_dataset_path = input('Enter unlabelled dataset path: ')
    #labelled_metadatasetpath = input('Enter the Labelled metadataset path: ')
    #unlabelled_metadataset_path = input('Enter the Unlabelled metadataset path')
    kmeans = KMeans(c)
    #metadata = Metadata(metadatapath='Data/labelled_set1.csv')
    #Unlabelledimagesdop_dict = metadata.getimagesdop_dict()

    label_features = LabelFeatures(labelled_dataset_path=labelled_dataset_path,
                                   unlabelled_dataset_path=unlabelled_dataset_path)
    label_features.set_features()
    dorsal_features = label_features.get_label_features('dorsal')
    #print(len(dorsal_features))
    palmar_features = label_features.get_label_features('palmar')
    #print(len(palmar_features))
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
    count = 0
    correct = 0
    for image_id in list(unlabelled_features.keys()):
        if similarity_val1[image_id] <= similarity_val2[image_id]:
            result[image_id]='dorsal'
            if count<50:
                correct+=1
        else:
            result[image_id]='palmar'
            if count>=50:
                correct+=1
        count+=1
    print(result)
    print('Accuracy:', correct/count)

elif task == '3':
    test_dataset_path = input("Enter test dataset path: ")




