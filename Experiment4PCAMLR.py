#Loren Anderson

#PCA with Linear Regression

import MatrixReader
import MatrixCreator
import DimensionReducer
import CrossVal
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

class PCAMLR:
    def __init__(self, train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type):
        self.train = train
        self.train_returns = train_returns
        self.test = test
        self.test_returns = test_returns
        self.freq_type = freq_type
        self.stopwords = stopwords
        self.svals = svals
        self.reduce_type = reduce_type
        self.results = None

    def run(self):
        mat_creat = MatrixCreator.MatrixCreator(self.train, self.freq_type, self.stopwords)
        new_train = mat_creat.get_count_matrix()
        new_test = mat_creat.transform_new_data(self.test)

        dim_reducer = DimensionReducer.DimensionReducer(new_train)
        reduced_train = dim_reducer.reduce_dimension(self.svals, self.reduce_type)
        reduced_test = dim_reducer.reduce_more(new_test.todense())

        cross_val = CrossVal.CrossVal(reduced_train, self.train_returns, reduced_test, self.test_returns, 2)
        final_folds = cross_val.get_decile_indices(self.test_returns)

        scores = self.predict(reduced_train, self.train_returns, reduced_test, final_folds)
        self.interpret_scores(scores)

    def predict(self, reduced_train, train_returns, reduced_test, final_folds):
        scores = []
        #lm = LinearRegression()
        #lm = RandomForestRegressor(n_estimators=100)
        lm = KNeighborsRegressor(n_neighbors=3, weights='distance')
        lm.fit(reduced_train, train_returns)
        print(type(train_returns[0]))
        for i in range(len(final_folds)):
            fold_stuff = np.take(reduced_test, final_folds[i], axis=0)
            vals = lm.predict(fold_stuff)
            scores.append(vals)
        return scores

    def interpret_scores(self, scores):
        data = []
        for score_list in scores:
            data.append(np.median(score_list))
        print(data)
        plt.xlabel('Quintiles')
        plt.ylabel('Returns')
        plt.title('Linear Regression')
        plt.plot(data, '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
        plt.show()



