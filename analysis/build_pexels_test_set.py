import pandas

from pathlib import Path
from os import listdir
from os.path import isfile, join

input_dataframe = pandas.read_csv('pexels_test_set.csv', delim_whitespace=True)

print(input_dataframe)

input_dataframe = input_dataframe.sample(n=1000)

input_dataframe.to_csv("./pexels_test_1000_set.csv", sep=',', index=False)