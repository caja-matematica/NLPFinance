# Loren Anderson

# machine learning methods

import numpy as np
import matplotlib.pyplot as plt

import MatrixVectorizer
import DimensionReducer
import CrossValidation
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


class ML:
    def __init__(self, train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method):
        self.run(train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method)

    def run(self, train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method):
        print("RUNNING")
        vectorizer = MatrixVectorizer.Vectorizer(train_corpus, freq_type, stopwords)
        train_count_matrix = vectorizer.get_count_matrix()
        test_count_matrix = vectorizer.transform_new_data(test_corpus).todense()

        #train_count_matrix = self.our_tfidf(train_count_matrix)

        dim_reducer = DimensionReducer.Reducer(train_count_matrix)
        reduced_train_count_matrix = dim_reducer.reduce_dimension(svals, reduce_type)
        reduced_test_count_matrix = dim_reducer.reduce_more_data(test_count_matrix)

        cross_val = CrossValidation.CrossVal(test_returns, bin_number)
        bins = cross_val.get_bins()

        scores = self.fit_predict(reduced_train_count_matrix, train_returns, reduced_test_count_matrix, bins, method)
        self.interpret_scores(scores)

    def fit_predict(self, reduced_train_count_matrix, train_returns, reduced_test_count_matrix, bins, method):
        # fit
        if method == 'lr':
            lm = LinearRegression()
        elif method == 'knn':
            lm = KNeighborsRegressor(n_neighbors=3, weights='distance')
        else:
            lm = RandomForestRegressor(n_estimators=100)
        lm.fit(reduced_train_count_matrix, train_returns)

        # predict
        scores = []
        for the_bin in bins:
            decile = np.take(reduced_test_count_matrix, the_bin, axis=0)
            #decile = self.our_tfidf(decile)
            vals = lm.predict(decile)
            scores.append(vals)
        return scores

    def interpret_scores(self, scores):
        data = []
        for score_list in scores:
            data.append(np.mean(score_list))
        print(data)
        plt.xlabel('Actual Return Deciles')
        plt.ylabel('Predicted Avg. Returns')
        plt.title('Actual Return Deciles vs. Predicted Avg. Returns')
        plt.plot(data, '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
        plt.show()

    def our_tfidf(self, matrix):
        rows, cols = matrix.shape

        doc_lengths = []
        for i in range(rows):
            print(i)
            doc_lengths.append(np.sum(matrix[i, :]))

        num_unique = []
        for j in range(cols):
            print(j)
            count = 0
            for i in range(rows):
                if matrix[i,j]>0:
                    count += 1
            num_unique.append(count)

        for i in range(rows):
            print('row: ', i)
            for j in range(cols):
                if matrix[i,j] != 0:
                    matrix[i,j] = (1+np.log(matrix[i,j]))/(1+np.log(doc_lengths[i]))*np.log(rows/(num_unique[j]+0.0))

        return matrix







