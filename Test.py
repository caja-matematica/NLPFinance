#Loren Anderson

import Experiment4PCAMLR

train = [
    "python, tools",
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

freq_type = 'count'
stopwords = []
svals = 2
new_corpus = ["Python, tools","ubuntu, loren"]
reduce_type = 'pca'
folds = 2

experiment = Experiment4PCAMLR.PCAMLR(train, train_returns, test, test_returns, freq_type, stopwords, svals, reduce_type, folds)
experiment.run()