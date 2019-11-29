# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
        self.dataset = list()

    def generate_input_data(self, dorsal_map, palmar_map):
        dorsalFeatures = list()
        for im, feat in dorsal_map.items():
            f = list()
            feat = list(feat)
            feat.append(0)
            dorsalFeatures.append(feat)

        palmarFeatures = list()

        for im, feat in palmar_map.items():
            f = list()
            feat = list(feat)
            feat.append(1)
            palmarFeatures.append(feat)

        self.dataset = dorsalFeatures
        for i in range(len(palmarFeatures)):
            self.dataset.append(palmarFeatures[i])

        return

    def fit(self, dataset):
        dfData = pd.DataFrame(dataset)
#        dfData = dfData.iloc[1:, :]
        y = dfData.iloc[:, -1]
        print(y.shape)
        XMain = dfData.iloc[:, :-1]
        X = XMain.iloc[:, :]
#        X_test = XMain.iloc[250:300,:]
        print("X & X test")
        print(X.shape)
#        print(X_test.shape)

        X = X.to_numpy()
        y = y.to_numpy()
        
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        A = cvxopt.matrix(A, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
#        print(solution['x'])
        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
