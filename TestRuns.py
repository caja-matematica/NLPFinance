#Created by Loren Anderson

# runs the tests

import csv
import CSVSizeReader
import CSVCountReader
import CSVReturnReader
import MLExperiments
import NegativeWordsExperiments
import FileSizeExperiments
import numpy as np


########################################################################################################################


def file_size_experiment():
    file1 = 'data/2016_returns.csv'
    file2 = 'data/2015_returns.csv'
    file3 = 'data/2014_returns.csv'
    file4 = 'data/2013_returns.csv'

    return_reader1 = CSVReturnReader.ReturnReader(file1)
    return_matrix1 = return_reader1.get_return_matrix()
    sorted_return_matrix1 = (return_matrix1[return_matrix1[:, 0].argsort()])[:, 4].astype(np.float)

    return_reader2 = CSVReturnReader.ReturnReader(file2)
    return_matrix2 = return_reader2.get_return_matrix()
    sorted_return_matrix2 = (return_matrix2[return_matrix2[:, 0].argsort()])[:, 4].astype(np.float)

    return_reader3 = CSVReturnReader.ReturnReader(file3)
    return_matrix3 = return_reader3.get_return_matrix()
    sorted_return_matrix3 = (return_matrix3[return_matrix3[:, 0].argsort()])[:, 4].astype(np.float)


    return_reader4 = CSVReturnReader.ReturnReader(file4)
    return_matrix4 = return_reader4.get_return_matrix()
    sorted_return_matrix4 = (return_matrix4[return_matrix4[:, 0].argsort()])[:, 4].astype(np.float)

    sorted_return_matrix = np.concatenate([sorted_return_matrix1,np.concatenate([sorted_return_matrix2, np.concatenate([sorted_return_matrix3,sorted_return_matrix4])])])

    file5 = 'data/2016_sizes.csv'
    file6 = 'data/2015_sizes.csv'
    file7 = 'data/2014_sizes.csv'
    file8 = 'data/2013_sizes.csv'

    size_reader1 = CSVSizeReader.SizeReader(file5)
    size_list1 = size_reader1.get_size_list()

    size_reader2 = CSVSizeReader.SizeReader(file6)
    size_list2 = size_reader2.get_size_list()

    size_reader3 = CSVSizeReader.SizeReader(file7)
    size_list3 = size_reader3.get_size_list()

    size_reader4 = CSVSizeReader.SizeReader(file8)
    size_list4 = size_reader4.get_size_list()

    size_list = size_list1 + size_list2 + size_list3 + size_list4


    #Parameters
    bin_number = 20

    experiment = FileSizeExperiments.FileSize(sorted_return_matrix, size_list, bin_number)


########################################################################################################################


def negative_word_experiment():
    file1 = 'data/2013_returns.csv'
    file2 = 'data/2013_words.csv'
    file3 = 'NegativeWordList.txt'

    return_reader1 = CSVReturnReader.ReturnReader(file1)
    count_reader_2 = CSVCountReader.CountReader(file2)

    #Parameters
    train_corpus = count_reader_2.get_word_list()
    freq_type = 'count'
    stopwords = []
    bin_number = 5
    train_returns = return_reader1.get_return_matrix()[:, 2].astype(np.float)

    experiment = NegativeWordsExperiments.NegFin(train_corpus, freq_type, stopwords, train_returns, bin_number, file3)


########################################################################################################################


def machine_learning_experiment():
    file1 = 'data/2015_returns.csv'
    file2 = 'data/2015_words.csv'
    file3 = 'data/2016_returns.csv'
    file4 = 'data/2016_words.csv'

    return_reader1 = CSVReturnReader.ReturnReader(file1)
    count_reader_2 = CSVCountReader.CountReader(file2)
    return_reader3 = CSVReturnReader.ReturnReader(file3)
    count_reader_4 = CSVCountReader.CountReader(file4)


    #Parameters
    train_corpus = count_reader_2.get_word_list()
    freq_type = 'tfidf'
    stopwords = []
    test_corpus = count_reader_4.get_word_list()
    svals  = 25
    reduce_type = 'lsa'
    test_returns = return_reader3.get_return_matrix()[:, 2].astype(np.float)
    bin_number = 5
    train_returns = return_reader1.get_return_matrix()[:, 2].astype(np.float)
    method = 'knn'

    experiment = MLExperiments.ML(train_corpus, freq_type, stopwords, test_corpus, svals, reduce_type, test_returns, bin_number, train_returns, method)


########################################################################################################################

file_size_experiment()
#negative_word_experiment()
#machine_learning_experiment()
