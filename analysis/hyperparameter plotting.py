import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


mean_distances = pd.read_csv(Path('sets') / Path('mean_distances.csv'))
mean_distances.drop(columns=['Unnamed: 0'], inplace=True)

print(mean_distances)

sns.set_theme()
hyperparameter_plots = True

if hyperparameter_plots:
    sns.relplot(
        data=mean_distances,
        x="mean_distance_to_orig", y="optim_lr"
    )
    plt.show()

    '''
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

distances_distorted_can = pd.read_csv(Path('sets/hyperparameter/distances_distorted_can.csv'))
distances_ssiae = pd.read_csv(Path('sets/hyperparameter/distances_ssiae.csv'))

data = [distances_ssiae['image_id'], distances_distorted_can["distance_to_orig"], distances_ssiae["distance_to_orig"],
        -(distances_ssiae["distance_to_orig"] - distances_distorted_can["distance_to_orig"])]
headers = ["image_id", "distance_orig_dist", "distance_orig_rec", "recovery_change"]

plot = pd.concat(data, axis=1, keys=headers)

html = plot.to_html()
with open(Path('results/distances_recovery_change_ssiae.html'), 'w') as file:
    file.write(html)

g = sns.scatterplot(
    data=plot,
    x='distance_orig_dist', y='recovery_change',
)
g.set(xlabel="SSIM distance between original and distorted image", ylabel="Change of SSIM after SSIAE was applied")
plt.ylim(-1, 1)

plt.show()

distances_ssiae = pd.read_csv(Path('results/hyperparametersearch/0.0/combined_data/hypersearch-1-32.csv'))

data = [distances_ssiae['image_id'], distances_distorted_can["distance_to_orig"], distances_ssiae["distance_to_orig"],
        -(distances_ssiae["distance_to_orig"] - distances_distorted_can["distance_to_orig"])]
headers = ["image_id", "distance_orig_dist", "distance_orig_rec", "recovery_change"]

plot = pd.concat(data, axis=1, keys=headers)

html = plot.to_html()
with open(Path('results/distances_recovery_change_ssiae.html'), 'w') as file:
    file.write(html)

g = sns.scatterplot(
    data=plot,
    x='distance_orig_dist', y='recovery_change',
)
g.set(xlabel="SSIM distance between original and distorted image", ylabel="Change of SSIM after SSIAE (new) was applied")
plt.ylim(-1, 1)

plt.show()

distances_nicer = pd.read_csv(Path('sets/hyperparameter/distances_nicer.csv'))
data = [distances_nicer['image_id'], distances_distorted_can["distance_to_orig"], distances_nicer["distance_to_orig"],
        -(distances_nicer["distance_to_orig"] - distances_distorted_can["distance_to_orig"])]
headers = ["image_id", "distance_orig_dist", "distance_orig_rec", "recovery_change"]

plot = pd.concat(data, axis=1, keys=headers)

html = plot.to_html()
with open(Path('results/distances_recovery_change_nicer.html'), 'w') as file:
    file.write(html)

g = sns.scatterplot(
    data=plot,
    x='distance_orig_dist', y='recovery_change',
)
g.set(xlabel="SSIM distance between original and distorted image", ylabel="Change of SSIM after NICER was applied")
plt.ylim(-1, 1)

plt.show()

print('Done!')