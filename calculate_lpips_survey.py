import os

import pandas as pd

from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from PIL import Image
from pathlib import Path
import lpips
import torch
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset_path = Path('datasets/survey')
original = dataset_path / 'original'
out = Path('analysis/sets/survey_lpips.csv')

styles = ['original', 'ssmtpiaa_sgd', 'ssmtpiaa_cma', 'nicer', 'expert']

images = [img.split('_')[0] for img in os.listdir(original)]
experts = os.listdir(dataset_path / 'expert')
originals = []
for img in os.listdir(original):
    print(img)
    originals.append(transform(Image.open(original / img)))
    print(len(originals))

results = pd.DataFrame(index=range(len(images)), columns=styles)

loss_fn_vgg = 0

for i in range(len(images)):
    if i % 10 == 0:
        print('del network')
        print(torch.cuda.memory_allocated())
        del loss_fn_vgg
        print(torch.cuda.memory_allocated())
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    for style in styles:
        if style != 'expert':
            img_tensor = transform(Image.open(dataset_path/style/f"{images[i]}_{style}.jpg")).to(device)
            results.at[i, style] = loss_fn_vgg.forward(originals[i].to(device), img_tensor)
            print('del tensor')
            print(torch.cuda.memory_allocated())
            del img_tensor
            del originals[i]
            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated())
        else:
            results.at[i, style] = loss_fn_vgg.forward(originals[i].to(device), transform(Image.open(dataset_path / style / experts[i])).to(device))
    torch.cuda.empty_cache()
    print(i)

print(results)
results.to_csv(out)