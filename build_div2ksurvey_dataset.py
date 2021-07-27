import pandas

from pathlib import Path
from os import listdir

photos = [f for f in listdir(Path('datasets/div2k/DIV2K_train_HR'))]
df = pandas.DataFrame(data={'image_id': photos})
print(df)
df = df.sample(n=500)
df.to_csv(Path('analysis/sets/div2ksurvey_base.csv'), sep=',', index=False)

print('Done!')