import time
import torch
import os
from argparse import ArgumentParser
from pathlib import Path

import config
from nicer import NICER
from analysis_utils import evaluate_distance_orig_distorted

# Init stuff
import sys
sys.path[0] = "."

start_time = time.time()

# Arguments from yaml

# Initialize output folders
filename = "distances_to_originals"
out = Path("analysis")
data = Path("results")

# Process images
evaluate_distance_orig_distorted(graph_data_path=out/data, filename=filename)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)