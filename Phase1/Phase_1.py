import sys
from features_images import FeaturesImages
from similar_images import Similarity
from DimRed import DimRed

task = input("Please specify the task number: ")
model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
if task == '1':
    image_id = input("Please specify the test image file name: ")
    features_image = FeaturesImages(model)
    features_image.compute_image_features(image_id, print_arr=True)

elif task == '2':
    folder_path = input("Please specify test folder path: ")
    features_folder = FeaturesImages(model, folder_path)
    features_folder.compute_features_images_folder()

elif task == '3':
    image_id = input("Please specify the test image file name: ")
    k = int(input("Please specify the value of K: "))
    test_dataset_path = input("Please specify test folder path: ")
    similarity = Similarity(model, image_id, k)
    similarity.get_similar_images(test_dataset_path)
    
elif task == '21':
    folder_path = input("Please specify test folder path: ")
    dmModel = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect model: ")
    k = int(input("Please specify the value of K: "))
    dimRed = DimRed(model, dmModel, k)
    dimRed.get_dimensionality_reduction(folder_path)
    
