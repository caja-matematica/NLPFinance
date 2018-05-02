#Loren Anderson

#creates matrix with returns
import numpy as np

class ReturnReader:
    def __init__(self, filename):
        self.filename = filename
        self.stuff = self.read_matrix()
        self.company_names = self.company_names()

    def read_matrix(self):
        import csv
        count = 0
        lists = []
        with open(self.filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if count != 0:
                    lists.append(row[1:len(row)-1])
                count += 1
        new_array = np.array(lists)
        return new_array

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

    def get_stuff(self):
        return self.stuff

    def get_company_names(self):
        return self.company_names