import config
import os
import torch
import time

from analysis_utils import edit_div2ksurvey
from nicer import NICER

start_time = time.time()

torch.Tensor([10, 1]).cuda()

print(os.getcwd())

output_file = 'ssmtpiaa_sgd_oldloss'

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

nicer.config.use_auto_brightness_normalizer = False
nicer.config.assessor = 'SSMTPIAA'
nicer.config.gamma = 0.053445881011986975
nicer.config.epochs = 50
nicer.config.optim = 'sgd'
nicer.config.optim_lr = 0.012797039203293
nicer.config.optim_momentum = 0.9
nicer.config.SSMTPIAA_loss = "COMPOSITE"
nicer.config.hinge_val = 0.15  # Margin
nicer.config.composite_pow = 0.8557224253386464
nicer.config.composite_balance = -0.7633380079031169  # Alpha
nicer.config.adaptive_score_offset = 0.12582415504241362

nicer.config.rescale = True
nicer.config.final_size = 1000

edit_div2ksurvey(nicer, output_file, nima_vgg16=False, nima_mobilenetv2=False, ssmtpiaa=True, ssmtpiaa_fine=False)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(elapsed_time)

