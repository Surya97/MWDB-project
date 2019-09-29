from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

'''
Base class for LDA decomposition.
'''

class LDAModel:
    def __init__(self, database_matrix, k_components):
        self.database_matrix = database_matrix
        self.k_components = k_components
        self.lda = LatentDirichletAllocation(n_components=self.k_components)
        self.principal_components = None
        self.decomposed_database_matrix = None

    '''
        This is the default method which does the decomposition.
    '''

    def decompose(self):
        self.principal_components = self.lda.fit_transform(self.database_matrix)
        # print(len(self.principal_components), len(self.principal_components[0]))
        self.decomposed_database_matrix = self.principal_components
        #print(self.decomposed_database_matrix)
        #print(len(self.lda.components_),len(self.lda.components_[0]))
        return
        #return self.print_term_weight_pairs()

    def get_feature_weight_values(self):
        return
        #return self.pca.explained_variance_

    def get_eigen_vectors(self):
        return np.array(self.principal_components).transpose()

    def get_decomposed_data_matrix(self):
        return self.decomposed_database_matrix

    def print_term_weight_pairs(self):
        eigen_values = self.get_feature_weight_values()
        eigen_vectors = self.get_eigen_vectors()
        count = 1
        for eigen_value, eigen_vector in zip(eigen_values, eigen_vectors):
            print("Latent feature", count)
            print("Eigen Value:", eigen_value)
            print("Eigen Vector:", eigen_vector)
            print()
            count += 1

    def get_data_latent_semantics(self):
        return self.decomposed_database_matrix




