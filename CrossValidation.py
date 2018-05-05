# Loren Anderson

# cross validation helper

import numpy as np


class CrossVal:
    def __init__(self, returns, number):
        self.bins = self.get_decile_indices(returns, number)

    def get_decile_indices(self, returns, number):
        # get bin counts
        bin_counts = np.zeros(number)
        for i in range(len(returns)):
            bin_counts[i % number] += 1

        # get bins
        bins = []
        sorted_returns = list(sorted(returns))
        current_index = 0
        for indices in bin_counts:
            bin = []
            for j in range(int(indices)):
                the_return = sorted_returns[current_index]
                for k in range(len(returns)):
                    if the_return == returns[k]:
                        bin.append(k)
                current_index += 1
            bins.append(bin)
        return bins

    def get_bins(self):
        return self.bins
