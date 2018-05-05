# Loren Anderson

# creates matrix with counts

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self, corpus, freq_type, stopwords):
        self.sparse_count_matrix, \
        self.word_list, \
        self.vectorizer = self.create_count_matrix(corpus, freq_type, stopwords)

    def create_count_matrix(self, corpus, freq_type, stopwords):
        if freq_type == 'count':
            vec = CountVectorizer(stop_words=stopwords)
            sparse_matrix = vec.fit_transform(corpus)
            word_list = vec.get_feature_names()
        elif freq_type == 'tfidf':
            vec = TfidfVectorizer(stop_words=stopwords)
            sparse_matrix = vec.fit_transform(corpus)
            word_list = vec.get_feature_names()
        return sparse_matrix, word_list, vec

    def transform_new_data(self, new_data):
        new_matrix = self.vectorizer.transform(new_data)
        return new_matrix

    def get_count_matrix(self):
        return self.sparse_count_matrix

    def get_word_list(self):
        return self.word_list
