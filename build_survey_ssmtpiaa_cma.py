import config
import os
import torch
import time

from analysis_utils import edit_survey
from nicer import NICER

start_time = time.time()

torch.Tensor([10, 1]).cuda()

print(os.getcwd())

output_file = 'ssmtpiaa_cma'

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

nicer.config.use_auto_brightness_normalizer = False
nicer.config.assessor = 'SSMTPIAA'
nicer.config.gamma = 0.1
nicer.config.epochs = 50
nicer.config.optim = 'cma'
nicer.config.cma_sigma = 0.4672401904167237
nicer.config.SSMTPIAA_loss = "COMPOSITE_NEW"
nicer.config.hinge_val = 0.1  # Margin
nicer.config.composite_new_balance = 0.3  # Alpha
nicer.config.adaptive_score_offset = 0.24045193277647922

edit_survey(nicer, output_file, nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)

