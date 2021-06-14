import pandas
import os
from dataset import Pexels
from PIL import Image

os.chdir(os.path.dirname(os.getcwd()))

pexels = Pexels(mode='pexels_test')

tall = []
wide = []

for i in range(len(pexels)):
    if i % 100 == 0:
        print(i)
    item = pexels.__getitem__(i)
    w, h = item['img'].size

    ratio = w / h

    if ratio >= 1.5:
        wide.append(item['image_id'])
    elif ratio <= 2/3:
        tall.append(item['image_id'])

wide_df = pandas.DataFrame({'images': wide})
print(wide_df)
wide_df = wide_df.sample(n=1000)
wide_df.to_csv("./analysis/sets/pexels_wide_set.csv", sep=',', index=False)

tall_df = pandas.DataFrame({'images': tall})
print(tall_df)
tall_df = tall_df.sample(n=1000)
tall_df.to_csv("./analysis/sets/pexels_tall_set.csv", sep=',', index=False)


