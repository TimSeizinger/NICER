import shutil
from pathlib import Path

import csv

with open(Path('analysis/sets/survey_base.csv'), newline='') as f:
    reader = csv.reader(f)
    image_ids = list(reader)

print(image_ids)

for image_id in image_ids:
    if image_id[0] == 'image_id':
        continue
    shutil.copy(Path(f'datasets/adobe5k/originals/{image_id[0]}.jpg'), Path(f'datasets/survey/original/{image_id[0]}_original.jpg'))