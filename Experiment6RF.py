#Loren Anderson

#PCA with Linear Regression

import MatrixCreator
import DimensionReducer
import CrossVal
from sklearn.ensemble import RandomForestRegressor

class RF:
    def __init__(self, train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds):
        self.train = train
        self.train_returns = train_returns
        self.test = test
        self.test_returns = test_returns
        self.freq_type = freq_type
        self.stopwords = stopwords
        self.svals = svals
        self.reduce_type = reduce_type
        self.folds = folds
        self.results = None

    def run(self):
        mat_creat = MatrixCreator.MatrixCreator(self.train, self.freq_type, self.stopwords)
        new_train = mat_creat.get_count_matrix()
        new_test = mat_creat.transform_new_data(self.test)

        dim_reducer = DimensionReducer.DimensionReducer(new_train)
        reduced_train = dim_reducer.reduce_dimension(self.svals, self.reduce_type)
        reduced_test = dim_reducer.reduce_more(new_test.todense())

        cross_val = CrossVal.CrossVal(reduced_train, self.train_returns, reduced_test, self.test_returns, self.folds)
        final_folds = cross_val.create_folds1()
        final_test, final_test_return = cross_val.shuffle_test()

        scores = self.predict(final_folds, final_test, final_test_return)
        print('Scores: ', scores)

    def predict(self, final_folds, final_test, final_test_return):
        scores = []
        for i in range(self.folds):
            lm = RandomForestRegressor()
            lm.fit(final_folds[i][0], final_folds[i][1])
            vals = lm.predict(final_test)
            print('Vals: ', vals)
            score = lm.score(final_test, final_test_return)
            scores.append(score)
        return scores