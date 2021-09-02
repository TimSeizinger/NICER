import config

from analysis_utils import evaluate_rating_pexels
from nicer import NICER

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)

evaluate_rating_pexels(nicer, 'pexels_rating_distribution', 'pexels_test_1000')