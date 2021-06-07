import numpy as np
import csv
import functools

# This information taken from the csv file 
bins = dict(zip([2,3,4,5,6,7,8,11],[1.5, 3, 4, 5, 6, 7, 9, 11]))

if __name__ == "__main__":

    with open('HCP_Test_Retest_Interval_Binned_Months_2017Mar31.csv', 'r') as f: 
        reader = csv.reader(f)
        next(reader)
        sub_bins = np.array([int(row[1]) for row in reader])

    total = functools.reduce(lambda acc,b: acc + bins[b], sub_bins)
    print("Mean time between sessions:", total / sub_bins.size)