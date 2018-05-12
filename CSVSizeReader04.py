# Loren Anderson

# constructs list of sizes from CSV file

import csv


class SizeReader:
    def __init__(self, filename):
        self.size_list = self.read_size_list(filename)

    def read_size_list(self, filename):
        sizes = []
        with open(filename) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                sizes.append(int(row[1]))
        return sizes

    def get_size_list(self):
        return self.size_list
