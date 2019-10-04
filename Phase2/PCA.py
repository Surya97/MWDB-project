from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
Base class for PCA decomposition.
'''


class PCAModel:
    def __init__(self, database_matrix, k_components):
        self.database_matrix = database_matrix
        self.k_components = k_components
        self.pca = PCA(n_components=self.k_components)
        self.principal_components = None
        self.decomposed_database_matrix = None

    '''
        This is the default method which does the decomposition.
    '''

    def decompose(self):
        scaled_feature_matrix = StandardScaler().fit_transform(self.database_matrix)
        self.principal_components = self.pca.fit_transform(scaled_feature_matrix)
        # print(len(self.principal_components), len(self.principal_components[0]))
        self.decomposed_database_matrix = self.principal_components
        return

    def get_feature_weight_values(self):
        return self.pca.explained_variance_

    def get_eigen_vectors(self):
        return np.array(self.principal_components).transpose()

    def get_decomposed_data_matrix(self):
        return self.decomposed_database_matrix

    def print_term_weight_pairs(self, k=-1):
        eigen_values = self.get_feature_weight_values()
        eigen_vectors = self.get_eigen_vectors()
        count = 1
        for eigen_value, eigen_vector in zip(eigen_values, eigen_vectors):
            if count>k:
                return
            print("Latent feature", count)
            print("Eigen Value:", eigen_value)
            print("Eigen Vector:", eigen_vector)
            print()
            count += 1

    def get_data_latent_semantics(self):
        return self.decomposed_database_matrix




