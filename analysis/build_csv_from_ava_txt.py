import pandas

from pathlib import Path
from os import listdir
from os.path import isfile, join

input_dataframe = pandas.read_csv(str(Path.home()) + '\Documents\datasets\images\AVA\AVA.txt', delim_whitespace=True)

print(input_dataframe)

dict = {'image_id': [], 'rating': []}
for index, row in input_dataframe.iterrows():
    total_rating = 0
    ratings = 0
    for i in range(1, 11):
        total_rating += row[str(i)] * i
        ratings += row[str(i)]
    dict['image_id'].append(row['img'])
    dict['rating'].append(total_rating/ratings)

output_dataframe = pandas.DataFrame(dict)

print(output_dataframe)

output_dataframe.to_csv("./ava_with_ratings.csv", sep=',', index=False)

top = output_dataframe.nlargest(1000, 'rating')

print(top)

top.to_csv("./ava_with_ratings_top.csv", sep=',', index=False)

bottom = output_dataframe.nsmallest(1000, 'rating')

print(bottom)

bottom.to_csv("./bottom_set.csv", sep=',', index=False)

print('Done!')