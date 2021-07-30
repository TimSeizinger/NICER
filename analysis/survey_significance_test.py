import pandas as pd
import numpy as np

from pathlib import Path
from survey_objects import style_data

import pandas as pd
import numpy as np
import survey_utils as utils
import random
import math

from pathlib import Path
import scipy.stats as stats
import scipy

print(scipy.__version__)

# Enables visualizations
visualization = True
relevant_columns = utils.get_relevant_columns_for_visualization(visualization)

# styles to evaluate further
styles = ['original', 'nicer', 'ssmtpiaa_sgd', 'ssmtpiaa_cma', 'expert']

# Load the datasets from both batches
survey1 = pd.read_csv(Path('sets/survey_results/processed/MTurk_Batch_1_approvals.csv'))
survey2 = pd.read_csv(Path('sets/survey_results/processed/MTurk_Batch_2_approvals.csv'))

# Combine the 2 suvey datasets and descramble ratings.
survey_result = utils.preprocess_data(survey1, survey2, visualization)

base_style = 'original'
enhanced_style = 'ssmtpiaa_sgd'

enhanced_is_better = []
random_list = []

for i in range(survey_result.shape[0]):
    best = 0
    # Get best rating for each image
    for style in [base_style, enhanced_style]:
        best = max(best, survey_result.at[i, f'{style}_rating'])
    # Find out to which style it belongs
    if survey_result.at[i, f'{enhanced_style}_rating'] == best:
        enhanced_is_better.append(1)

n = survey_result.shape[0]
k = len(enhanced_is_better)
p = 0.5

print(n)
print(k)

result = stats.binomtest(k, n, p, alternative='greater')

print(result.pvalue)
print("{:.0%}".format(result.pvalue))
print(result.proportion_ci())
print(result)

result = stats.binomtest(2, 10, 0.5, alternative='greater')

print("{:.0%}".format(result.pvalue))
print(result.proportion_ci())
print(result)
