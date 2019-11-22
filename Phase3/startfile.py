from customlshash import LSHash
import pickle
import sys



lsh = LSHash(6, 40,5, matrices_filename ='test.npz')
final_path = '../Phase2/pickle_files/LBP_PCA.pkl'
print('loading from pickle file path', final_path)
infile = open(final_path, 'rb')
dataset_features = pickle.load(infile)
val = None

for image_id, feature in dataset_features.items():
    lsh.index(feature,image_id=image_id)

ret_val =lsh.query(dataset_features['Hand_0009128.jpg'],image_id='Hand_0009132.jpg')

for val in ret_val:
    print(val[1])