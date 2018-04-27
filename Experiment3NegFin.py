#PCA with Linear Regression

import MatrixCreator
import DimensionReducer
import CrossVal
import numpy as np

class NegFin:
    def __init__(self, train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds, filename):
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
        self.filename = filename
        self.word_list = None

    def run(self):
        mat_creat = MatrixCreator.MatrixCreator(self.train, self.freq_type, self.stopwords)
        new_train = mat_creat.get_count_matrix()

        self.create_negative_stuff()
        new_neg = mat_creat.transform_new_data(self.word_list)
        new_neg = new_neg.todense()

        cross_val = CrossVal.CrossVal(new_train.todense(), self.train_returns, [], [], self.folds)
        final_folds = cross_val.create_folds1()

        scores = []
        for i in range(len(final_folds)):
            fold = final_folds[i][0]
            guess = np.matmul(new_neg,np.transpose(fold))
            target = final_folds[i][1]
            guess = guess.flatten()
            target = target.flatten()
            new_guess = []
            new_target = []
            row1,col1 = guess.shape
            row2,col2 = target.shape
            for j in range(col1):
                new_guess.append(guess[0,j])
            for j in range(col2):
                new_target.append(target[0,j])
            print(new_guess)
            print(new_target)
            score = self.score(new_guess, new_target)
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

    def create_negative_stuff(self):
        file = open(self.filename, 'r')
        word_array = file.readlines()
        word_list = ""
        for word in word_array:
            word_length = len(word)
            word_list += word[0:word_length-1].lower() + ", "
        word_list = word_list[0:len(word_list)-1]
        self.word_list = [word_list]

    #def score(self, targets, predictions):

