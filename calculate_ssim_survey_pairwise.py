import os
import itertools

import pandas as pd

from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from PIL import Image
from pathlib import Path

dataset_path = Path('datasets/survey')

original = dataset_path / 'original'
sgd = dataset_path / 'ssmtpiaa_sgd'
cma = dataset_path / 'ssmtpiaa_cma'
nicer = dataset_path / 'nicer'
expert = dataset_path / 'expert'

out = Path('analysis/sets/survey_similarities_pairwise.csv')

styles = ['original', 'ssmtpiaa_sgd', 'ssmtpiaa_cma', 'nicer', 'expert']
pairs_to_evaluate = list(itertools.combinations(styles, 2))

images = [img.split('_')[0] for img in os.listdir(original)]
experts = os.listdir(dataset_path / 'expert')

imagedict = {'original': [], 'ssmtpiaa_sgd': [], 'ssmtpiaa_cma': [], 'nicer': [], 'expert': []}

for style in styles:
    for img in os.listdir(dataset_path / style):
        imagedict[style].append(dataset_path / style / img)

results = pd.DataFrame(index=styles, columns=styles)

for pair in pairs_to_evaluate:
    print(pair)
    results.at[pair[0], pair[1]] = []
    results.at[pair[1], pair[0]] = []
    for i in range(len(images)):
        imgssim = ssim(img_as_float(Image.open(imagedict[pair[0]][i])), img_as_float(Image.open(imagedict[pair[1]][i])), multichannel=True)
        results.at[pair[0], pair[1]].append(imgssim)
        results.at[pair[1], pair[0]].append(imgssim)
        print(i)

print(results)
results.to_csv(out)