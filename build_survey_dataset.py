import pandas

from pathlib import Path
from os import listdir

photos = [f.split('.')[0] for f in listdir(Path('datasets/adobe5k/originals'))]
df = pandas.DataFrame(data={'image_id': photos})
print(df)
df = df.sample(n=500)
df.to_csv(Path('analysis/sets/survey_base.csv'), sep=',', index=False)

print('Done!')