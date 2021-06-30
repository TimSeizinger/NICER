import pandas

from pathlib import Path
from os import listdir

photos = [str(f[:len(f)-4]) for f in listdir(str(Path.home()) + '\\Documents\\datasets\\images\\pexels\\images\\')
          #if isfile(join((str(Path.home()) + '\\Documents\\datasets\\images\\pexels\\images\\'), f)) and f.endswith('.jpeg')
          ]
df = pandas.DataFrame(data={'images': photos})
df.to_csv("./pexels.csv", sep=',', index=False)

print('Done!')