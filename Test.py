#Loren Anderson

import Experiment4PCAMLR
import Experiment3NegFin
import Experiment2FileSize
import Experiment5NN
import Experiment6RF
import MatrixReader
import MatrixCreator

train = [
    "python, tools, abandon",
    "linux, tools, ubuntu",
    "distributed systems, linux, networking, tools",
    "python, ubuntu, linux, networking"
]

test = [
    "python, linux",
    "distributed systems, networking"
]

train_returns = [[1],[-1],[1],[2]]
test_returns = [[1],[-1]]

train_filesizes = [[2],[-2],[1],[4]]

freq_type = 'tfidf'
stopwords = []
svals = 2
new_corpus = ["Python, tools","ubuntu, loren"]
reduce_type = 'lsa'
folds = 2
filename = 'NegativeWordList.txt'

#----------------------------------------
#-----       Experiment 4  --------------
#----------------------------------------

#experiment4 = Experiment4PCAMLR.PCAMLR(train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds)
#experiment4.run()

#----------------------------------------
#-----       Experiment 3  --------------
#----------------------------------------

#experiment3 = Experiment3NegFin.NegFin(train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds, filename)
#experiment3.run()

#----------------------------------------
#-----       Experiment 2  --------------
#----------------------------------------

#experiment2 = Experiment2FileSize.FileSize(train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds, filename, train_filesizes)
#experiment2.run()

#----------------------------------------
#-----       Experiment 5  --------------
#----------------------------------------

#experiment5 = Experiment5NN.KNN(train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds)
#experiment5.run()

#----------------------------------------
#-----       Experiment 6  --------------
#----------------------------------------

experiment6 = Experiment6RF.RF(train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds)
experiment6.run()

#----------------------------------------
#-----       Matrix Reader --------------
#----------------------------------------

#file = 'matrixstuff.csv'
#mat_read = MatrixReader.MatrixReader(file)
#matstuff = mat_read.get_stuff()
#print(matstuff)
#mat_creat = MatrixCreator.MatrixCreator(matstuff, 'count', [])
#new_train = mat_creat.get_count_matrix()
#print(new_train.shape)