from PIL import Image
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim

original = img_as_float(np.array(Image.open('out/xperiment/original.jpeg')))
print(original.shape)
distorted = img_as_float(np.array(Image.open('out/xperiment/distorted.jpeg')))
print(f"Similarity between original and distorted is {1-ssim(original, distorted, multichannel=True)}")
nima = distorted = img_as_float(np.array(Image.open('out/xperiment/n')))
