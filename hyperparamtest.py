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

# Arguments from yaml
parser = ArgumentParser()
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--optim_lr", type=float, required=True)
parser.add_argument("--gamma", type=float, required=True)
parser.add_argument("--score_pow", type=float, required=True)
parser.add_argument("--composite_balance", type=float, required=True)
parser.add_argument("--adaptive_score_offset", type=float, required=True)
args = parser.parse_args()

# Initialize output folders
folder = Path(f"{args.optim_lr}_{args.gamma}_{args.score_pow}_{args.composite_balance}_{args.adaptive_score_offset}")
out = Path("out")
data = Path("data")
if not os.path.isdir(out/folder):
    os.mkdir(out/folder)

if not os.path.isdir(out/folder/data):
    os.mkdir(out/folder/data)

# Initialize Data Loader and NICER
nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

# Set hyperparameters
nicer.config.optim_lr = args.optim_lr
nicer.config.gamma = args.gamma
nicer.config.composite_pow = args.score_pow
nicer.config.composite_balance = args.composite_balance
nicer.config.adaptive_score_offset = args.adaptive_score_offset

# Process images
evaluate_editing_recovery_pexels(nicer=nicer, sample_size=args.sample_size, img_path=out/folder,
                                 graph_data_path=out/folder/data, filename=folder, loss='Composite',
                                 nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)