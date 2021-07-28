import shutil
from pathlib import Path
import csv
import random

with open(Path('analysis/sets/survey_base.csv'), newline='') as f:
    reader = csv.reader(f)
    image_ids = list(reader)
    image_ids = [id[0] for id in image_ids]

print(image_ids)

for image_id in image_ids:
    rng = random.randint(0, 4)
    if rng == 0:
        shutil.copy(Path(f'datasets/adobe5k/experts/experta/{image_id}.jpg'),
                    Path(f'datasets/survey/expert/{image_id}_experta.jpg'))
    elif rng == 1:
        shutil.copy(Path(f'datasets/adobe5k/experts/expertb/{image_id}.jpg'),
                    Path(f'datasets/survey/expert/{image_id}_expertb.jpg'))
    elif rng == 2:
        shutil.copy(Path(f'datasets/adobe5k/experts/expertc/{image_id}.jpg'),
                    Path(f'datasets/survey/expert/{image_id}_expertc.jpg'))
    elif rng == 3:
        shutil.copy(Path(f'datasets/adobe5k/experts/expertd/{image_id}.jpg'),
                    Path(f'datasets/survey/expert/{image_id}_expertd.jpg'))
    elif rng == 4:
        shutil.copy(Path(f'datasets/adobe5k/experts/experte/{image_id}.jpg'),
                    Path(f'datasets/survey/expert/{image_id}_experte.jpg'))