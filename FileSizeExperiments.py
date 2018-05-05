# Loren Anderson

# file size

import MatrixVectorizer
import CrossValidation
import numpy as np
import matplotlib.pyplot as plt

class FileSize:
    def __init__(self, train_returns, train_sizes, bin_number):
        self.run(train_returns, train_sizes, bin_number)

    def run(self, train_returns, train_sizes, bin_number):
        cross_val = CrossValidation.CrossVal(train_returns, bin_number)
        bins = cross_val.get_bins()

        scores = self.predict(train_sizes, bins)
        self.interpret_scores(scores)

    def predict(self, train_sizes, final_folds):
        final_scores = []
        for fold in final_folds:
            small_list = []
            for index in fold:
                small_list.append(int(train_sizes[index]))
            final_scores.append(np.median(small_list))
        return final_scores

    def interpret_scores(self, scores):
        data = []
        for score_list in scores:
            data.append(np.mean(score_list))
        plt.xlabel('Deciles')
        plt.ylabel('Avg Filesize')
        plt.title('Filesize')
        plt.plot(data, '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
        plt.show()