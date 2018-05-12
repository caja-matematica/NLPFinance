# Loren Anderson

# creates list of lists of words in each 10-K file

import csv


class CountReader:
    def __init__(self, filename):
        self.word_list = self.read_matrix(filename)
        self.company_names = self.company_names(filename)

    # creates count matrix
    def read_matrix(self, filename):
        lists = []
        f = open(filename)
        with f as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                the_list = self.get_list(row)
                lists.append(the_list)
        return lists

    # helper method that gets list of words in a single 10-K
    def get_list(self, row):
        the_string = ""
        words = row.keys()
        for key in words:
            if key != "":
                number = row.get(key)
                if number == '':
                    number = 0
                else:
                    number = int(float(number))
                for j in range(number):
                    the_string += key + ", "
        new_string = the_string[0:len(the_string)-2]  # remove final ", "
        return new_string

    # gets list of company names in matrix
    def company_names(self, filename):
        names = []
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                names.append(row.get(""))
        all_names = names[1:len(names)]
        return all_names

    def get_word_list(self):
        return self.word_list

    def get_company_names(self):
        return self.company_names
