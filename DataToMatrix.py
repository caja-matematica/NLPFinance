#Loren Anderson

# creates matrix with counts

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class DataToMatrix:
    def __init__(self, corpus, freq_type, stopwords):
        self.sparse_count_matrix, self.word_list = self.create_count_matrix(corpus, freq_type, stopwords)

    def create_count_matrix(self, corpus, freq_type, stopwords):
        if freq_type == 'count':
            vec = CountVectorizer(stop_words=stopwords)
            sparse_matrix = vec.fit_transform(corpus)
            word_list = vec.get_feature_names()
        elif freq_type == 'tfidf':
            vec = TfidfVectorizer(stop_words=stopwords)
            sparse_matrix = vec.fit_transform(corpus)
            word_list = vec.get_feature_names()
        return sparse_matrix,word_list

corpus = [
  "python, tools",
  "linux, tools, ubuntu",
  "distributed systems, linux, networking, tools",
]
freq_type = 'count'
stopwords = []

sparse_matrix = DataToMatrix(corpus=corpus, freq_type=freq_type, stopwords=stopwords).sparse_count_matrix
print(sparse_matrix)