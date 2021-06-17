import os
import threading
import config

from analysis_utils import evaluate_rating_pexels
from nicer import NICER

pwd = os.getcwd()
os.chdir(os.path.dirname(os.getcwd()))

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

evaluate_wide = threading.Thread(target=evaluate_rating_pexels, args=(nicer, 'pexels_wide', 'pexels_wide'))
evaluate_wide.start()
evaluate_tall = threading.Thread(target=evaluate_rating_pexels, args=(nicer, 'pexels_tall', 'pexels_tall'))
evaluate_tall.start()
'''
evaluate_pexels(nicer, 'pexels_wide', 'pexels_wide')
evaluate_pexels(nicer, 'pexels_tall', 'pexels_tall')
'''