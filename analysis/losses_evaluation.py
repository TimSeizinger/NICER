import pandas as pd
import numpy as np
import survey_utils as utils
import random
import os
import math

from pathlib import Path
from IPython.core.display import display, HTML
import scipy

print(f"Scipy version is: {scipy.__version__}, needs to be 1.7.0 or greater")

print(os.path.abspath(os.curdir))

# Enables visualizations
visualization = True

# Load the datasets from both batches
scores = pd.read_csv(Path('results/pexels_rating_distribution.csv'))

print('Done!')

print(np.mean(scores["orig_ia_pre_score"]))
print(np.mean(scores["orig_ia_pre_styles_change_strength"]))

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

ax = sns.displot(scores, x="orig_ia_pre_score", binwidth=0.05)
plt.axvline(np.mean(scores["orig_ia_pre_score"]), color='red', label='mean')
plt.legend()
ax.set(xlabel="Ranking Score")
ax.savefig(Path('results/rankingscore_distribution_pexels.svg'))

matplotlib.rcParams['text.usetex'] = True

ax = sns.displot(scores, x="orig_ia_pre_styles_change_strength", binwidth=0.05)
plt.axvline(np.mean(scores["orig_ia_pre_styles_change_strength"]), color='red', label='mean')
plt.legend()
plt.xlim(0, 1)
ax.set(xlabel="$s_{regr}(\vec{v})$")
ax.savefig(Path('results/regressionscore_distribution_pexels.svg'))