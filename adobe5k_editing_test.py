import config
import os
import torch
import time
from argparse import ArgumentParser

from analysis_utils import evaluate_editing_adobe5k
from nicer import NICER

start_time = time.time()

torch.Tensor([10, 1]).cuda()

parser = ArgumentParser()
parser.add_argument("--output", type=str, required=False)
parser.add_argument("--limit", type=int, required=False)
parser.add_argument("--mode", type=str, required=False)
args = parser.parse_args()

if args.limit is not None:
    output = str(args.output)
    limit: int = int(args.limit)
    mode: str = str(args.mode)
else:
    output = 'adobe5k_editing_test'
    limit = 200
    mode = 'ssmtpiaa'

'''
pwd = os.getcwd()
os.chdir(os.path.dirname(os.getcwd()))
'''

print(os.getcwd())

#output_file = 'adobe5k_editing_test'
output_file = output

if not os.path.isdir("./analysis/results/"):
    os.mkdir("./analysis/results/")

if not os.path.isdir("./analysis/results/" + output_file + "/"):
    os.mkdir("./analysis/results/" + output_file + "/")

if not os.path.isdir("./analysis/results/" + output_file + '_graph_data' + "/"):
    os.mkdir("./analysis/results/" + output_file + '_graph_data' + "/")

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

if mode == 'NIMA':
    # Set hyperparameters
    nicer.config.assessor = 'NIMA_VGG16'
    nicer.config.use_auto_brightness_normalizer = True
    nicer.config.gamma = 0.1
    nicer.config.optim_lr = 0.05

    evaluate_editing_adobe5k(nicer, output_file, 'InputAsShotZeroed',
                               nima_vgg16=True, nima_mobilenetv2=False, ssmtpiaa=False, ssmtpiaa_fine=False, limit=limit)
else:
    evaluate_editing_adobe5k(nicer, output_file, 'InputAsShotZeroed',
                               nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False, limit=limit)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)


'''
evaluate_pexels(nicer, 'pexels_wide', 'pexels_wide')
evaluate_pexels(nicer, 'pexels_tall', 'pexels_tall')
'''