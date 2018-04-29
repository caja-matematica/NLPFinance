#Loren Anderson

# creates matrix with counts

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class MatrixReader:
    def __init__(self, filename):
        self.filename = filename
        self.word_list = self.word_list()
        self.stuff = self.read_matrix()
        self.company_names = self.company_names()

    def word_list(self):
        import csv
        word_list = []
        with open(self.filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                word_list.append(row)
                break
        new_word_list = word_list[1:len(word_list)]
        return new_word_list

    def read_matrix(self):
        import csv
        count = 0
        lists = []
        with open(self.filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if count != 0:
                    the_list = self.get_list(row)
                    lists.append(the_list)
                count += 1
        return lists

    def get_list(self, row):
        the_string = ""
        words = row.keys()
        for key in words:
            if key != "":
                number = int(row.get(key))
                for j in range(number):
                    the_string += key + ", "
        new_string = the_string[0:len(the_string)-2]
        return new_string

    def company_names(self):
        import csv
        names = []
        with open(self.filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                names.append(row.get(""))
        all_names = names[1:len(names)]
        return all_names

    def get_filename(self):
        return self.filename

    def get_word_list(self):
        return self.word_list

    def get_stuff(self):
        return self.stuff

    def get_company_names(self):
        return self.company_names