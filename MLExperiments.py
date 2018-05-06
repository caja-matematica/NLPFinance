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

class ML:
    def __init__(self, train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method):
        self.run(train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method)

    def run(self, train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method):
        vectorizer = MatrixVectorizer.Vectorizer(train_corpus, freq_type, stopwords)
        train_count_matrix = vectorizer.get_count_matrix()
        test_count_matrix = vectorizer.transform_new_data(test_corpus)

        dim_reducer = DimensionReducer.Reducer(train_count_matrix)
        reduced_train_count_matrix = dim_reducer.reduce_dimension(svals, reduce_type)
        reduced_test_count_matrix = dim_reducer.reduce_more_data(test_count_matrix.todense())

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
            vals = lm.predict(decile)
            scores.append(vals)
        return scores

    def interpret_scores(self, scores):
        data = []
        for score_list in scores:
            data.append(np.mean(score_list))
        print(data)
        plt.xlabel('Deciles')
        plt.ylabel('Returns')
        plt.title('Linear Regression')
        plt.plot(data, '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
        plt.show()



