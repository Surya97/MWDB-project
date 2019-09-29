import site
site.addsitedir('../Phase1')

from Decomposition import Decomposition

task = input("Please specify the task number: ")
decomposition = input("1.PCA\n2.SVD\n3.NMF\n4.LDA\nSelect decomposition: ")
model = input("1.CM\n2.LBP\n3.HOG\n4.SIFT\nSelect model: ")
if task == '1':
    k = int(input("Enter the number of latent features to consider: "))
    folder_path = input("Please specify test folder path: ")
    decomposition = Decomposition(decomposition, k, model, folder_path)
    decomposition.dimensionality_reduction()
