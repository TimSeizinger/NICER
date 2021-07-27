import config
import os
import torch
import time

from analysis_utils import edit_div2ksurvey
from nicer import NICER

start_time = time.time()

torch.Tensor([10, 1]).cuda()

print(os.getcwd())

output_file = 'nicer'

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

nicer.config.use_auto_brightness_normalizer = True
nicer.config.assessor = 'NIMA_VGG16'
nicer.config.gamma = 0.1
nicer.config.epochs = 50
nicer.config.optim = 'sgd'
nicer.config.optim_lr = 0.05
nicer.config.optim_momentum = 0.9

nicer.config.rescale = True
nicer.config.final_size = 1000

edit_div2ksurvey(nicer, output_file, nima_vgg16=True, nima_mobilenetv2=False, ssmtpiaa=False,
                               ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)

