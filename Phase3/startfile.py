from customlshash import MyCustomLSH
import pickle
import sys



lsh = MyCustomLSH(6,40,5)
final_path = '../Phase2/pickle_files/LBP_PCA.pkl'
print('loading from pickle file path', final_path)
infile = open(final_path, 'rb')
dataset_features = pickle.load(infile)
val = None

for image_id, feature in dataset_features.items():
    lsh.add_to_index_structure(feature, image_id=image_id)

ret_val =lsh.query(dataset_features['Hand_0009132.jpg'],image_id='Hand_0009132.jpg')

for val in ret_val:
    print(val[1])