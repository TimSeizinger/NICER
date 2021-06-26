import config
import os
import torch
import time

from analysis_utils import evaluate_editing_pexels
from nicer import NICER

start_time = time.time()

torch.Tensor([10, 1]).cuda()

'''
pwd = os.getcwd()
os.chdir(os.path.dirname(os.getcwd()))
'''

print(os.getcwd())

output_file = 'pexels_wide_edit_score'

if not os.path.isdir("./analysis/results/"):
    os.mkdir("./analysis/results/")

if not os.path.isdir("./analysis/results/" + output_file + "/"):
    os.mkdir("./analysis/results/" + output_file + "/")

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

evaluate_editing_pexels(nicer, output_file, 'pexels_wide', nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False, limit=10)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)


'''
evaluate_pexels(nicer, 'pexels_wide', 'pexels_wide')
evaluate_pexels(nicer, 'pexels_tall', 'pexels_tall')
'''