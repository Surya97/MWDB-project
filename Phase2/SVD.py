from numpy.linalg import svd
import numpy as np


class SVD:
    def __init__(self, database_matrix, k_components):
        self.database_matrix = database_matrix
        self.k_components = k_components
        self.u = None
        self.vh = None
        self.s = None
        self.reduced_database_matrix = None

    def decompose(self):
        self.u, self.s, self.vh = svd(self.database_matrix, full_matrices=False)
        # print('Original database matrix dimensions', len(self.database_matrix), len(self.database_matrix[0]))
        self.get_decomposed_data_matrix()
        return self.print_term_weight_pairs()

    def get_eigen_vectors(self):
        return self.vh[:self.k_components]

    def get_feature_weight_values(self):
        return self.s[:self.k_components]

    def get_decomposed_data_matrix(self):
        self.reduced_database_matrix = np.dot(self.u[:, :self.k_components], np.diag(self.s[:self.k_components]))
        # print(len(self.reduced_database_matrix), len(self.reduced_database_matrix[0]))
        # for i in range(len(self.reduced_database_matrix)):
        #     print(self.reduced_database_matrix[i])
        return self.reduced_database_matrix

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




