import sys
from features_images import FeaturesImages
from similar_images import Similarity


task = input("Please specify the task number: ")
model = input("1.LBP\n2.HOG\nSelect model: ")
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

# arguments = sys.argv[1:]
#
# task = arguments[0]
#
# model = arguments[1]
#
# if task == '1':
#     image_id = arguments[2]
#     features_image = FeaturesImages(model)
#     features_image.compute_image_features(image_id, print_arr=True)
#
# elif task == '2':
#     folder_path = arguments[2]
#     features_folder = FeaturesImages(model, folder_path)
#     features_folder.compute_features_images_folder()
# if task == '3':
#     image_id = arguments[2]
#     k = int(arguments[3])
#     test_dataset_path = arguments[4]
#     similarity = Similarity(model, image_id, k)
#     similarity.get_similar_images(test_dataset_path)

