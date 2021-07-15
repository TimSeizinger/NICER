import time
import torch
import os
from argparse import ArgumentParser
from pathlib import Path

import config
from nicer import NICER
from analysis_utils import evaluate_editing_recovery_pexels_new

# Init stuff
import sys
sys.path[0] = "."

torch.Tensor([10, 1]).cuda()
start_time = time.time()

# Arguments from yaml
parser = ArgumentParser()
parser.add_argument("--optim_lr", type=float, required=True)
parser.add_argument("--margin", type=float, required=True)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--gamma", type=float, required=True)
parser.add_argument("--adaptive_score_offset", type=float, required=True)
args = parser.parse_args()

#Convert args to their datatypes
optim_lr: float = float(args.optim_lr)
gamma: float = float(args.gamma)
margin: float = float(args.margin)
alpha: float = float(args.alpha)
adaptive_score_offset: float = float(args.adaptive_score_offset)

# Initialize output folders
folder = Path(f"{optim_lr}_{gamma}_{margin}_{alpha}_{adaptive_score_offset}")
out = Path("out")
data = Path("data")
if not os.path.isdir(out/folder):
    os.mkdir(out/folder)

if not os.path.isdir(out/folder/data):
    os.mkdir(out/folder/data)

# Initialize NICER
nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

# Set hyperparameters
nicer.config.optim_lr = optim_lr
nicer.config.gamma = gamma
nicer.config.hinge_val = margin
nicer.config.composite_new_balance = alpha
nicer.config.adaptive_score_offset = adaptive_score_offset


# Process images
evaluate_editing_recovery_pexels_new(nicer=nicer, img_path=out/folder,
                                 graph_data_path=out/folder/data, filename=folder, loss='COMPOSITE_NEW',
                                 nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)