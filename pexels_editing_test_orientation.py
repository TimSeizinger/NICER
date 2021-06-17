import config
import os

from analysis_utils import evaluate_editing_pexels
from nicer import NICER

'''
pwd = os.getcwd()
os.chdir(os.path.dirname(os.getcwd()))
'''

output_file = 'pexels_wide_edit_score'

if not os.path.isdir("./analysis/results/" + output_file + "/"):
    os.mkdir("./analysis/results/" + output_file + "/")

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

evaluate_editing_pexels(nicer, 'output_file', 'pexels_wide', limit=10)

'''
evaluate_pexels(nicer, 'pexels_wide', 'pexels_wide')
evaluate_pexels(nicer, 'pexels_tall', 'pexels_tall')
'''