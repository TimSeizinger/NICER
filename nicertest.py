import time
import torch
import os
from argparse import ArgumentParser
from pathlib import Path

import config
from nicer import NICER
from analysis_utils import evaluate_editing_recovery_pexels

# Init stuff
import sys
sys.path[0] = "."

torch.Tensor([10, 1]).cuda()
start_time = time.time()


# Initialize output folders
folder = Path(f"Nicer")
out = Path("out")
data = Path("data")
if not os.path.isdir(out/folder):
    os.mkdir(out/folder)

if not os.path.isdir(out/folder/data):
    os.mkdir(out/folder/data)

# Initialize NICER
nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

# Set hyperparameters
nicer.config.assessor = 'NIMA_VGG16'
nicer.config.use_auto_brightness_normalizer = True
nicer.config.gamma = 0.1
nicer.config.optim_lr = 0.05


# Process images
evaluate_editing_recovery_pexels(nicer=nicer, img_path=out/folder,
                                 graph_data_path=out/folder/data, filename=folder, loss='COMPOSITE',
                                 nima_vgg16=True, nima_mobilenetv2=False, ssmtpiaa=False, ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)