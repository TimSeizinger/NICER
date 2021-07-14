import time
import torch
import os
from pathlib import Path

import config
from nicer import NICER
from analysis_utils import can_test

# Init stuff
import sys
sys.path[0] = "."

torch.Tensor([10, 1]).cuda()
start_time = time.time()

# Initialize output folders
folder = Path(f"can_test_orig")
out = Path("out")
data = Path("data")
if not os.path.isdir(out/folder):
    os.mkdir(out/folder)

if not os.path.isdir(out/folder/data):
    os.mkdir(out/folder/data)

# Initialize NICER
nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

# Set hyperparameters
nicer.config.epochs = 0


# Process images
can_test(nicer=nicer, img_path=out/folder, graph_data_path=out/folder/data, filename=folder, mode='orig')

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)