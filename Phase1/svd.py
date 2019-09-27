import numpy as np
from features_images import FeaturesImages
from pathlib import Path
import misc
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import collections


def get_latent_semantics_pair(a,b,k):
    k_latent_semantics={}
    for i in range(k):
        k_latent_semantics[a[i]] = b[i]
    return k_latent_semantics

def SVD_MODEL(feature_vectors, k):
    print('SVD:')
    uChannel, sChannel, vhChannel = np.linalg.svd(feature_vectors)

    k_latent_semantics=get_latent_semantics_pair(sChannel,uChannel,k)
    for first,second in k_latent_semantics.items():
        print(first, second)

def PCA_MODEL(feature_vectors, k):
    print('PCA:')
    x = StandardScaler().fit_transform(feature_vectors)
    pca = PCA(n_components=k)
    principalComponents = pca.fit_transform(x)
    #pca = PCA()
    #pca.fit(feature_vectors)
    #doubt:should scale or not
    k_latent_semantics=get_latent_semantics_pair(pca.explained_variance_, principalComponents,k)

    for first,second in k_latent_semantics.items():
        print(first, second)

def NMF_MODEL(feature_vectors, k):
    print('NMF:')
    nmf=NMF(n_components=5)
    nmf.fit(feature_vectors)

    print(nmf.components_)


if __name__ == '__main__':
    dataset_images_features = misc.load_from_pickle(os.path.dirname(__file__), 'CM')
    ranking = {}
    feature_vectors = []
    for image_id, feature_vector in tqdm(dataset_images_features.items()):
        feature_vectors.append(feature_vector)
    #LATENT FEATURES EXTRACTIONS:

    #SVD_MODEL(feature_vectors, 5)
    PCA_MODEL(feature_vectors, 5)
    #NMF_MODEL(feature_vectors, 5)

