# Loren Anderson

# dimensionality reduction on the count matrix

import matplotlib.pyplot as plt
import sklearn.decomposition.pca
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.random_projection import GaussianRandomProjection


class Reducer:
    def __init__(self, sparse_count_matrix):
        self.sparse_count_matrix = sparse_count_matrix
        self.reduced = sparse_count_matrix
        self.reducer = None

    def reduce_dimension(self, svals, reduce_type):
        if reduce_type == 'pca':
            reducer = sklearn.decomposition.pca.PCA(n_components=svals)
            reduced = reducer.fit_transform(self.sparse_count_matrix.todense())
        elif reduce_type == 'lsa':
            reducer = sklearn.decomposition.TruncatedSVD(n_components=svals)
            reduced = reducer.fit_transform(self.sparse_count_matrix)
        elif reduce_type == 'nmf':
            reducer = NMF(n_components=svals, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
            reduced = reducer.fit_transform(self.sparse_count_matrix)
        elif reduce_type == 'lda':
            reducer = LatentDirichletAllocation(n_topics=svals, max_iter=5, learning_method='online',
                                            learning_offset=50., random_state=0)
            reduced = reducer.fit_transform(self.sparse_count_matrix)
        elif reduce_type == 'grp':
            reducer = GaussianRandomProjection(n_components=svals)
            reduced = reducer.fit_transform(self.sparse_count_matrix)
        else:
            reduced = self.sparse_count_matrix
            reducer = None
        self.reducer = reducer
        self.reduced = reduced
        return reduced

    def reduce_more_data(self, more_data):
        more_reduced = self.reducer.transform(more_data)
        return more_reduced

    def return_reduced(self):
        return self.reduced

    def analyze_pca(self):
        pca = sklearn.decomposition.pca.PCA()
        pca.fit_transform(self.sparse_count_matrix.todense())
        variance = pca.explained_variance_
        variance_ratio = pca.explained_variance_ratio_

        # Begin Scree Plot
        plt.figure(1)
        # Values
        plt.subplot(121)
        plt.plot(variance)
        plt.title('Singular Values')
        plt.xlabel('Top-n singular values')
        plt.ylabel('Numerical Value')
        # Percent Variance
        plt.subplot(122)
        plt.plot(variance_ratio)
        plt.title('Percent Variance Captured')
        plt.xlabel('Dimensions')
        plt.ylabel('Explained Variance')
        #Show Plot
        plt.show()
