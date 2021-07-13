import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


mean_distances = pd.read_csv(Path('sets') / Path('mean_distances.csv'))
mean_distances.drop(columns=['Unnamed: 0'], inplace=True)

print(mean_distances)

sns.set_theme()
'''
sns.relplot(
    data=mean_distances,
    x="mean_distance_to_orig", y="optim_lr"
)
plt.show()


sns.relplot(
    data=mean_distances,
    x="mean_distance_to_orig", y="gamma"
)
plt.show()

sns.relplot(
    data=mean_distances,
    x="mean_distance_to_orig", y="score_pow"
)
plt.show()

sns.relplot(
    data=mean_distances,
    x="mean_distance_to_orig", y="composite_balance"
)
plt.show()

sns.relplot(
    data=mean_distances,
    x="mean_distance_to_orig", y="adaptive_score_offset"
)
plt.show()
'''



deep_pal = sns.color_palette('deep')
cmap = sns.blend_palette([deep_pal[2], deep_pal[1]], as_cmap=True)

#mean_distances = mean_distances.sample(n=10)

sns.pairplot(
    data=mean_distances,
    hue='mean_distance_to_orig',
)

plt.show()

g = sns.histplot(
    data=mean_distances,
    x='optim_lr', y='adaptive_score_offset',
    hue='mean_distance_to_orig',
    palette=cmap,
)

plt.legend([],[], frameon=False)

plt.show()

g = sns.histplot(
    data=mean_distances,
    x='composite_balance', y='adaptive_score_offset',
    hue='mean_distance_to_orig',
    palette=cmap,
)

plt.legend([],[], frameon=False)

plt.show()


print('Done!')