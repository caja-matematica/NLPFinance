#Loren Anderson

#Cross Validation Helper

import numpy as np

class CrossVal:
    def __init__(self, train, train_returns, test, test_returns, folds):
        self.train = train
        self.train_returns = train_returns
        self.test = test
        self.test_returns = test_returns
        self.folds = folds

    def create_folds1(self):
        train_data = np.concatenate((self.train_returns,self.train), axis = 1)
        np.random.shuffle(train_data)

        rows, cols = train_data.shape
        choice = []
        for i in range(self.folds):
            choice.append(0)

        for i in range(rows):
            index = int(i % self.folds)
            choice[index] += 1

        list_of_lists = []
        for i in range(self.folds):
            small_list = []
            for j in range(choice[i]):
                small_list.append(i*self.folds+j)
            list_of_lists.append(small_list)

        list_of_matrices = []
        for i in range(self.folds):
            reduced_matrix = np.delete(train_data, list_of_lists[i], axis=0)
            reduced_train = reduced_matrix[:, 1:]
            reduced_train_returns = reduced_matrix[:, 0]
            list_of_matrices.append([reduced_train, reduced_train_returns])
        return list_of_matrices

    def shuffle_test(self):
        test_data = np.concatenate((self.test_returns,self.test), axis = 1)
        np.random.shuffle(test_data)
        shuffled_test = test_data[:, 1:]
        shuffled_test_return = test_data[:, 0]
        return shuffled_test, shuffled_test_return
