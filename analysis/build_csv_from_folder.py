import pandas

from pathlib import Path
from os import listdir
from os.path import isfile, join

photos = [int(f[:len(f)-4]) for f in listdir(str(Path.home()) + '\Documents\datasets\images\Landscapes\images') if
          isfile(join((str(Path.home()) + '\Documents\datasets\images\Landscapes\images'), f)) and f.endswith('.jpg')]
df = pandas.DataFrame(data={'images': photos})
df.to_csv("./landscapes_set.csv", sep=',', index=False)

print('Done!')