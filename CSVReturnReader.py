# Loren Anderson

# constructs matrix of returns from csv file

import numpy as np
import csv


class ReturnReader:
    def __init__(self, filename):
        self.return_matrix = self.read_matrix(filename)

    # returns array containing company names and returns
    def read_matrix(self, filename):
        count = 0
        lists = []
        with open(filename) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if count != 0:
                    lists.append(row[0:len(row)-1])
                count += 1
        new_array = np.array(lists)
        return new_array

    # returns the return matrix
    def get_return_matrix(self):
        return self.return_matrix
