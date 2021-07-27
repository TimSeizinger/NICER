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

start_time = time.time()

#Convert args to their datatypes
sigma: float = float(5)
gamma: float = float(0.1)
margin: float = float(0.15)
alpha: float = float(0.5)
adaptive_score_offset: float = float(0.3)

# Initialize output folders
folder = Path(f"{sigma}_{gamma}_{margin}_{alpha}_{adaptive_score_offset}")
out = Path("out_cma")
data = Path("data")
if not os.path.isdir(out/folder):
    os.mkdir(out/folder)

if not os.path.isdir(out/folder/data):
    os.mkdir(out/folder/data)

# Initialize NICER
nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path, device='cpu')

# Set hyperparameters
nicer.config.optim = 'cma'
nicer.config.cma_sigma = sigma
nicer.config.gamma = gamma
nicer.config.hinge_val = margin
nicer.config.composite_new_balance = alpha
nicer.config.adaptive_score_offset = adaptive_score_offset


# Process images
evaluate_editing_recovery_pexels(nicer=nicer, img_path=out/folder,
                                 graph_data_path=out/folder/data, filename=folder, loss='COMPOSITE',
                                 nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)
