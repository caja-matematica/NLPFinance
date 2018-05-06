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

        document_scores = self.score_documents(negative_word_matrix, train_count_matrix)

        cross_val = CrossValidation.CrossVal(document_scores, bin_number)
        bins = cross_val.get_bins()

        scores = self.predict(train_returns, bins)
        self.interpret_scores(scores)
        print(scores)

    def score_documents(self, negative_word_matrix, train_count_matrix):
        negative_word_matrix = np.array(negative_word_matrix.todense()).flatten()
        train_count_matrix = np.array(train_count_matrix.todense())
        #document_scores = np.matmul(train_count_matrix, np.transpose(negative_word_matrix))
        #document_scores = document_scores.flatten()

        #get negative word matrix indices
        negative_indices = []
        for i in range(len(negative_word_matrix)):
            if negative_word_matrix[i] > 0:
                negative_indices.append(i)

        rows, cols = train_count_matrix.shape
        N = rows

        #get document frequencies
        frequencies = []
        for index in negative_indices:
            count = 0
            for i in range(rows):
                if train_count_matrix[i, index] > 0:
                    count += 1
            frequencies.append(count)

        #get# unique words in each document
        unique_words = []
        for row in range(N):
            count = 0
            for i in range(cols):
                if train_count_matrix[row, i] > 0:
                    count += 1
            unique_words.append(count)

        final_scores = []
        for row in range(N):
            doc_score = 0
            a = np.sum(train_count_matrix[row, :])/(unique_words[row])
            for j in range(len(negative_indices)):
                term_count = train_count_matrix[row, negative_indices[j]]
                if term_count != 0:
                    term_score = (1.0 + np.log(term_count))/(1.0+np.log(a))*N/frequencies[j]
                    doc_score -= term_score
            final_scores.append(doc_score)
        return final_scores

    def predict(self, train_returns, bins):
        scores = []
        for the_bin in bins:
            small_list = []
            for index in the_bin:
                small_list.append(float(train_returns[index]))
            scores.append(np.mean(small_list))
        return scores

    def interpret_scores(self, scores):
        data = []
        for score_list in scores:
            data.append(np.mean(score_list))
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
