from numpy.linalg import svd
import numpy as np


class SVD:
    def __init__(self, database_matrix, k_components):
        self.database_matrix = database_matrix
        self.k_components = k_components
        self.u = None
        self.vh = None
        self.s = None
        self.decomposed_database_matrix = None

    def decompose(self):
        self.u, self.s, self.vh = svd(self.database_matrix, full_matrices=False)
        # print(len(self.u), len(self.u[0]))
        # print(len(self.s))
        # print(len(self.vh), len(self.vh[0]))
        self.decomposed_database_matrix = np.matmul(np.matmul(self.u, np.diag(self.s)), self.vh)
        #print(len(self.decomposed_database_matrix), len(self.decomposed_database_matrix[0]))
        return self.print_term_weight_pairs()

    def get_eigen_vectors(self):
        return self.vh[:self.k_components]

    def get_feature_weight_values(self):
        return self.s[:self.k_components]

    def get_decomposed_data_matrix(self):
        return self.decomposed_database_matrix

    def print_term_weight_pairs(self):
        eigen_vectors = self.get_eigen_vectors()
        eigen_values = self.get_feature_weight_values()
        count = 1
        for eigen_value, eigen_vector in zip(eigen_values, eigen_vectors):
            print("Latent feature", count)
            print("Eigen Value:", eigen_value)
            print("Eigen Vector:", eigen_vector)
            print()
            count += 1




