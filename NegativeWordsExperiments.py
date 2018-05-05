# Loren Anderson

# Negative Financial Word Proportion

import MatrixVectorizer
import CrossValidation
import numpy as np
import matplotlib.pyplot as plt


class NegFin:
    def __init__(self, train_corpus, freq_type, stopwords, train_returns, bin_number, filename):
        self.run(train_corpus, freq_type, stopwords, train_returns, bin_number, filename)

    def run(self, train_corpus, freq_type, stopwords, train_returns, bin_number, filename):
        vectorizer = MatrixVectorizer.Vectorizer(train_corpus, freq_type, stopwords)
        train_count_matrix = vectorizer.get_count_matrix()

        negative_word_list = self.create_negative_stuff(filename)
        negative_word_matrix = vectorizer.transform_new_data(negative_word_list)

        cross_val = CrossValidation.CrossVal(train_returns, bin_number)
        bins = cross_val.get_bins()

        scores = self.predict(train_count_matrix, negative_word_matrix, bins)
        self.interpret_scores(scores)

    def predict(self, train_count_matrix, negative_word_matrix, bins):
        negative_word_matrix = np.array(negative_word_matrix.todense())
        train_count_matrix = np.array(train_count_matrix.todense())
        scores = np.matmul(train_count_matrix, np.transpose(negative_word_matrix))
        scores = scores.flatten()
        final_scores = []
        for bin in bins:
            small_list = []
            for index in bin:
                small_list.append(scores[index]/np.sum(train_count_matrix[index, :]))
            final_scores.append(np.mean(small_list))
        return final_scores

    def interpret_scores(self, scores):
        data = []
        for score_list in scores:
            data.append(np.median(score_list))
        plt.xlabel('Deciles')
        plt.ylabel('Average Negative Words')
        plt.title('Method 3: Word Tones')
        plt.plot(data, '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
        plt.show()

    def create_negative_stuff(self, filename):
        file = open(filename, 'r')
        word_array = file.readlines()
        word_list = ""
        for word in word_array:
            word_length = len(word)
            word_list += word[0:word_length-1].lower() + ", "
        word_list = word_list[0:len(word_list)-1]
        return [word_list]
