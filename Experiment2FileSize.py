#Loren Anderson

#PCA with Linear Regression

import MatrixCreator
import DimensionReducer
import CrossVal
import numpy as np

class FileSize:
    def __init__(self, train, train_sizes, test, test_returns, freq_type, stopwords, svals, reduce_type, folds, filename, train_filesizes):
        self.train = train
        self.train_returns = train_sizes
        self.test = test
        self.test_returns = test_returns
        self.freq_type = freq_type
        self.stopwords = stopwords
        self.svals = svals
        self.reduce_type = reduce_type
        self.folds = folds
        self.results = None
        self.filename = filename
        self.word_list = None
        self.train_filesizes = train_filesizes

    def run(self):
        mat_creat = MatrixCreator.MatrixCreator(self.train, self.freq_type, self.stopwords)
        new_train = mat_creat.get_count_matrix()

        cross_val = CrossVal.CrossVal(self.train_filesizes, self.train_returns, [], [], self.folds)
        final_folds = cross_val.create_folds1()

        scores = []
        for i in range(len(final_folds)):
            guess = final_folds[i][0]
            target = final_folds[i][1]
            guess = guess.flatten()
            target = target.flatten()
            score = self.score(guess, target)
            scores.append(score)
        print('Scores: ', scores)

    def get_ranks(self, vals):
        sorted_vals = sorted(vals)
        ranks = []
        for val in vals:
            for i in range(len(vals)):
                if val == sorted_vals[i]:
                    ranks.append(i)
                    break
        return ranks

    def score(self, guess, target):
        guess_ranks = self.get_ranks(guess)
        target_ranks = self.get_ranks(target)
        sum = 0
        for i in range(len(guess_ranks)):
            sum += ((guess_ranks[i]-target_ranks[i])**2)
        return sum