import pandas as pd
from pathlib import Path

df = pd.read_csv(Path('analysis')/Path('results')/'distances_to_originals.csv')
print(df['distance_to_orig'].mean())
print(df['distance_to_orig'].std())

'''
from PIL import Image
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import lpips
import torchvision.transforms as transforms

original = img_as_float(np.array(Image.open('out/xperiment/original.jpg')))
print(original.shape)
distorted = img_as_float(np.array(Image.open('out/xperiment/distorted.jpg')))
print(f"SSIM based Similarity between original and itself is {ssim(original, original, multichannel=True)}")
print(f"SSIM based Similarity between original and distorted is {ssim(original, distorted, multichannel=True)}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(512),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

original = transform(Image.open('out/xperiment/original.jpg'))
distorted = transform(Image.open('out/xperiment/distorted.jpg'))

print("processed images for LPIPS")

loss_fn_alex = lpips.LPIPS(net='vgg')

print(f"LPIPS vgg based Similarity between original and itself is {1 - loss_fn_alex(original, original).item()}")
print(f"LPIPS vgg based Similarity between original and distorted is {1 - loss_fn_alex(original, distorted).item()}")

s_fn_alex = lpips.LPIPS(net='alex')

print(f"LPIPS alex based Similarity between original and itself is {1 - loss_fn_alex(original, original).item()}")
print(f"LPIPS alex based Similarity between original and distorted is {1 - loss_fn_alex(original, distorted).item()}")
'''