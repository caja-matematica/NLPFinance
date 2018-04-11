# Creates a matrix storing the total daily return (with div) data
# Allows quick access to monthly returns

import csv
import numpy as np


class DailyReturn:
    def __init__(self):
        print("Initialized")

    # gets the number of rows in csv file f
    def get_num_rows(self, filename):
        f = open(filename)
        csv_f = csv.reader(f)
        rows = 0
        for row in csv_f:
            rows += 1
        return rows

    # gets the number of columns in csv file f
    def get_num_cols(self, filename):
        f = open(filename)
        csv_f = csv.reader(f)
        cols = 0
        for row in csv_f:
            cols = len(row)
            break
        return cols

    # creates an identical table of the csv file with first row removed
    def create_table(self, filename):
        num_rows = self.get_num_rows(filename)
        num_cols = self.get_num_cols(filename)
        table = np.zeros((num_rows, num_cols), dtype=object)  # initialize table

        # append rows of csv file to table
        f = open(filename)
        csv_f = csv.reader(f)
        count = 0
        for row in csv_f:
            table[count] = row
            count += 1
        table = np.delete(table, 0, 0)  # delete the first row of the csv file
        return table

    # get list of key value pairs of company name and row index
    def get_company_pairs(self, table):
        rows = len(table)
        pairs = [[table[0,2],0]]  # initialize
        for i in range(rows):
            if i != rows-1:
                if table[i][2] != table[i+1][2]:
                    pairs.append([table[i+1][2], i+1])
        return pairs

    # get starting index of a ticker symbol in the table
    def get_ticker_starting_index(self, pairs, ticker):
        for i in range(len(pairs)):
            if ticker == pairs[i][0]:
                return pairs[i][1]
        return 0

    # gets the index in the table of a ticker of with a date
    def get_ticker_date_index(self, pairs, ticker, table, date):
        starting_index = self.get_ticker_starting_index(pairs, ticker)
        rows = len(table)
        for i in range(starting_index, rows):
            if date <= int(table[i][1]):
                return i

    # gets the year, month, and day of date
    def get_year_month_day(self, date):
        day = int(date % 100)
        month = int(((date % 10000) - day)/100)
        year = int((date - 100*month - day)/10000)
        return (year, month,day)

    # adds a certain number of months to a date, from 1 to 12
    def add_months_to_date(self, date, months):
        year, month, day = self.get_year_month_day(date)
        if month + months > 12:
            year += 1
        month = month+months % 12
        return int(year*10000+month*100+day)

    # calculates a return
    def get_return(self, table, ticker, date, months):
        pairs = self.get_company_pairs(table)
        end_date = self.add_months_to_date(date, months)
        first_index = self.get_ticker_date_index(pairs, ticker, table, date)
        second_index = self.get_ticker_date_index(pairs, ticker, table, end_date)

        the_return = 1.0
        for i in range(first_index,second_index+1):
            the_return *= (float(table[i][4])+1)
        return the_return - 1

    #test file
    def output_stuff(self):

        # inputs
        filename = 'Total.Daily.Return.(with.div).csv'
        ticker = 'PLXS'
        date = 20150104
        months = 1

        # calculations
        table = self.create_table(filename)                        # only needs to be created once!
        the_return = self.get_return(table, ticker, date, months)  # can loop through ticker, date, months, etc.
        print(the_return)

        # manually check above calculation
        new_return = 1
        for i in range(1, 23):
            new_return *= (float(table[i][4])+1)
        print(new_return - 1)

dr = DailyReturn()
dr.output_stuff()
