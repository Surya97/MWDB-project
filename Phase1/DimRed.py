from features_images import FeaturesImages
import os
from pathlib import Path
import misc
from tqdm import tqdm
import collections
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DimRed:
    def __init__(self, model_name, dim_model_name, k):
        self.model_name = model_name;
        self.dim_model_name = dim_model_name
        self.k = k

    def get_dimensionality_reduction(self, dataset_folder):
        dataSet = []
        features_images = FeaturesImages(self.model_name)
        dataset_folder_path = os.path.join(Path(os.path.dirname(__file__)).parent, dataset_folder)
        model = features_images.get_model()
        dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), self.model_name)
#        datasetDataFrame = pd.DataFrame(list(dataset_images_features.items()))
        for image_id, feature_vector in tqdm(dataset_images_features.items()):
            dataRow = []
            dataRow.append(image_id)
            for item in feature_vector:
                dataRow.append(item)
            dataSet.append(dataRow)
            with open('datasetDataFrame.csv', 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(dataRow)
            myfile.close()
        print(np.shape(dataSet))
#        print(type(dataset_images_features))
#        for image_id, feature_vector in tqdm(dataset_images_features.items()):
        x = pd.DataFrame(dataSet)
        x = x.iloc[:, 1:]
        X = StandardScaler().fit_transform(x)
        print(np.shape(X))
        
        
        pca = PCA(n_components=self.k)
        principalComponents = pca.fit_transform(X)
        print(np.shape(principalComponents))