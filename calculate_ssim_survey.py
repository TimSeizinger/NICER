import os

import pandas as pd

from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from PIL import Image
from pathlib import Path

dataset_path = Path('datasets/survey')
original = dataset_path / 'original'
out = Path('analysis/sets/survey_similarities.csv')

styles = ['original', 'ssmtpiaa_sgd', 'ssmtpiaa_cma', 'nicer', 'expert']

images = [img.split('_')[0] for img in os.listdir(original)]
experts = os.listdir(dataset_path / 'expert')
originals = []
for img in os.listdir(original):
    print(img)
    originals.append(img_as_float(Image.open(original / img)))
    print(len(originals))

results = pd.DataFrame(index=range(len(images)), columns=styles)

for i in range(len(images)):
    for style in styles:
        if style != 'expert':
            results.at[i, style] = ssim(originals[i], img_as_float(Image.open(dataset_path/style/f"{images[i]}_{style}.jpg")), multichannel=True)
        else:
            results.at[i, style] = ssim(originals[i],
                                        img_as_float(Image.open(dataset_path / style / experts[i])),
                                        multichannel=True)
    print(i)

print(results)
results.to_csv(out)