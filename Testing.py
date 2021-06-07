import numpy as np
from PIL import Image
from mdfloss import MDFLoss
import torch
import imageio

def get_tensor(img):
    img = torch.from_numpy(imageio.core.asarray(img / 255.0))
    img = img.type(dtype=torch.float64)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0).type(torch.cuda.FloatTensor)
    img.to(device)
    return img

loss_func = MDFLoss('models/Ds_Denoising.pth', torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

original = imageio.imread('out/loss_testing/original.jpg')


#modified = np.asarray(Image.open('out/loss_testing/sat-10.jpg'))
sat_30 = imageio.imread('out/loss_testing/sat-30.jpg')

sat_30_loclap = imageio.imread('out/loss_testing/sat-30_loclap-100.jpg')

sat_30_restored = imageio.imread('out/loss_testing/restored/sat-30+40.jpg')

sat_30_restored_loclap = imageio.imread('out/loss_testing/restored/sat-30+40_loclap-100.jpg')

print('sat_30: ', str((np.square(original - sat_30)).mean(axis=0).mean(axis=0).mean(axis=0)))

print('sat_30_loclap: ', str(np.square(original - sat_30_loclap).mean(axis=0).mean(axis=0).mean(axis=0)))

print('sat-30+40: ', str((np.square(original - sat_30_restored)).mean(axis=0).mean(axis=0).mean(axis=0)))

print('sat-30+40_loclap: ', str((np.square(original - sat_30_restored_loclap)).mean(axis=0).mean(axis=0).mean(axis=0)))

original = get_tensor(original)

sat_30 = get_tensor(sat_30)

sat_30_restored = get_tensor(sat_30_restored)

sat_30_loclap = get_tensor(sat_30_loclap)

sat_30_restored_loclap = get_tensor(sat_30_restored_loclap)


print('sat_30: ', str(loss_func(original, sat_30).item()))

print('sat_30_loclap: ', str(loss_func(original, sat_30_loclap).item()))

print('sat-30+40: ', str(loss_func(original, sat_30_restored).item()))

print('sat-30+40_loclap: ', str(loss_func(original, sat_30_restored_loclap).item()))

