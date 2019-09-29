import sys
sys.path.insert(1, '../Phase1')

from similar_images import Similarity

from Decomposition import Decomposition

task = input("Please specify the task number: ")
model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
decomposition = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")

if task == '1':
    k = int(input("Enter the number of latent features to consider: "))
    folder_path = input("Please specify test folder path: ")
    decomposition = Decomposition(decomposition, k, model, folder_path)
    decomposition.dimensionality_reduction()

elif task == '2':
    image_id = input("Please specify the test image file name: ")
    m = int(input("Please specify the value of m: "))
    test_dataset_path = input("Please specify test folder path: ")
    decomposition = Decomposition(decomposition, m, model, test_dataset_path)
    similarity = Similarity(model, image_id, m, decomposition)
    similarity.get_similar_images(test_dataset_path, reduced_dimension=True)
else:
    print('Please enter the correct task number !')
