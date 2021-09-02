import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

from pathlib import Path

outpath = Path('results/hyperparameterplots')

xsize = 6
ysize = 5

show = True

mean_distances = pd.read_csv(Path('sets') / Path('mean_distances_new.csv'))
mean_distances.drop(columns=['Unnamed: 0'], inplace=True)



for i in range(mean_distances.shape[0]):
    mean_distances.at[i, 'mean_similarity_to_original'] = 1 - mean_distances.at[i, 'mean_distance_to_orig']
mean_distances.drop(columns=['mean_distance_to_orig'], inplace=True)

top_mean_distances = mean_distances.nlargest(int(mean_distances.shape[0]*0.25), 'mean_similarity_to_original')
bottom_mean_distances = mean_distances.nsmallest(int(mean_distances.shape[0]*0.75), 'mean_similarity_to_original')
bottom25_mean_distances = mean_distances.nsmallest(int(mean_distances.shape[0]*0.25), 'mean_similarity_to_original')

top_mean_distances.drop(columns=['yaml_name', 'mean_similarity_to_original'], inplace=True)


sns.set_theme()
hyperparameter_plots = False

deep_pal = sns.color_palette('deep')
cmap = sns.blend_palette([deep_pal[1], deep_pal[2]], as_cmap=True)

if hyperparameter_plots:
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
    

    sns.pairplot(
        data=mean_distances,
    )

    plt.show()

    sns.pairplot(
        data=mean_distances,
        hue='mean_similarity_to_original',
    )
    
    plt.show()
    '''

    mean_distances_rank = mean_distances[mean_distances['alpha'] >= 0.8]

    mean_distances_rank_offset = mean_distances_rank[mean_distances_rank['adaptive_score_offset'] <= 0.3]

    g = sns.regplot(
        data=mean_distances,
        x='optim_lr', y='mean_similarity_to_original',
    )

    sns.scatterplot(
        data=mean_distances_rank_offset,
        x='optim_lr', y='mean_similarity_to_original',
        color="r", marker="P"
    )

    plt.xlim(0, 0.25)
    g.set(ylabel="mean SSIM between original and recovered images",
          xlabel="Hyperparameter eta")

    plt.savefig(outpath / 'eta.svg')
    if show:
        plt.show()

    print(f'Pearson Correlation of eta with performance: {stats.pearsonr(mean_distances["optim_lr"], mean_distances["mean_similarity_to_original"])}')

    g = sns.regplot(
        data=mean_distances,
        x='gamma', y='mean_similarity_to_original',
    )
    g.set(ylabel="mean SSIM between original and recovered images",
          xlabel="Hyperparameter gamma")

    plt.savefig(outpath / 'gamma.svg')
    if show:
        plt.show()

    print(f'Pearson Correlation of gamma with performance: {stats.pearsonr(mean_distances["gamma"], mean_distances["mean_similarity_to_original"])}')

    g = sns.regplot(
        data=mean_distances,
        x='adaptive_score_offset', y='mean_similarity_to_original',
    )

    sns.scatterplot(
        data=mean_distances_rank,
        x='adaptive_score_offset', y='mean_similarity_to_original',
        color="r", marker="P"
    )

    g.set(ylabel="mean SSIM between original and recovered images",
          xlabel="Hyperparameter delta")
    plt.xlim(0, 1)

    plt.savefig(outpath / 'delta.svg')
    if show:
        plt.show()

    print(f'Pearson Correlation of delta with performance: {stats.pearsonr(mean_distances["adaptive_score_offset"], mean_distances["mean_similarity_to_original"])}')

    #Filter for low influence of this parameter

    g = sns.regplot(
        data=mean_distances_rank,
        x='adaptive_score_offset', y='mean_similarity_to_original',
    )
    g.set(ylabel="mean SSIM between original and recovered images",
          xlabel="Hyperparameter delta")
    plt.xlim(0, 1)

    plt.savefig(outpath / 'delta_regr.svg')
    if show:
        plt.show()

    #print(f'Pearson Correlation of delta with performance: {stats.pearsonr(mean_distances_rank_offset["adaptive_score_offset"], mean_distances_rank_offset["mean_similarity_to_original"])}')

    mean_distances_mu = mean_distances[mean_distances['alpha'] <= 0.2]

    g = sns.regplot(
        data=mean_distances,
        x='margin', y='mean_similarity_to_original',
    )

    sns.scatterplot(
        data=mean_distances_mu,
        x='margin', y='mean_similarity_to_original',
        color="r", marker="P"
    )

    g.set(ylabel="mean SSIM between original and recovered images",
          xlabel="Hyperparameter mu")

    plt.savefig(outpath / 'mu.svg')
    if show:
        plt.show()

    print(
        f'Pearson Correlation of mu with performance: {stats.pearsonr(mean_distances["margin"], mean_distances["mean_similarity_to_original"])}')

    print(
        f'Pearson Correlation of mu when alpha is small with performance: {stats.pearsonr(mean_distances_mu["margin"], mean_distances_mu["mean_similarity_to_original"])}')

    g = sns.regplot(
        data=mean_distances,
        x='alpha', y='mean_similarity_to_original',
    )
    g.set(ylabel="mean SSIM between original and recovered images",
          xlabel="Hyperparameter beta")

    plt.savefig(outpath / 'beta.svg')
    if show:
        plt.show()

    print(
        f'Pearson Correlation of beta with performance: {stats.pearsonr(mean_distances["alpha"], mean_distances["mean_similarity_to_original"])}')

    #Distplots


    g = sns.displot(
        data=top_mean_distances['optim_lr'],
        binwidth=0.025,
        aspect=1
    )

    g.set(xlabel="Value of eta")
    plt.xlim(0, 0.25)
    plt.savefig(outpath / 'eta_distribution.svg')
    if show:
        plt.show()

    print('eta')
    print(stats.ks_2samp(top_mean_distances['optim_lr'], bottom_mean_distances['optim_lr']))
    print(stats.mannwhitneyu(top_mean_distances['optim_lr'], bottom25_mean_distances['optim_lr']))

    g = sns.displot(
        data=top_mean_distances['gamma'],
        binwidth=0.025,
        aspect=1
    )

    g.set(xlabel="Value of gamma")

    plt.savefig(outpath / 'gamma_distribution.svg')
    if show:
        plt.show()

    print('gamma')
    print(stats.ks_2samp(top_mean_distances['gamma'], bottom_mean_distances['gamma']))
    print(stats.mannwhitneyu(top_mean_distances['gamma'], bottom25_mean_distances['gamma']))


    g = sns.displot(
        data=top_mean_distances['adaptive_score_offset'],
        binwidth=0.1,
        aspect=1,
    )
    plt.xlim(0, 1)
    g.set(xlabel="Value of delta")

    plt.savefig(outpath / 'delta_distribution.svg')
    if show:
        plt.show()

    print('adaptive_score_offset')
    print(stats.ks_2samp(top_mean_distances['adaptive_score_offset'], bottom_mean_distances['adaptive_score_offset']))
    print(stats.mannwhitneyu(top_mean_distances['adaptive_score_offset'], bottom25_mean_distances['adaptive_score_offset']))


    g = sns.displot(
        data=top_mean_distances['margin'],
        binwidth=0.0125,
        aspect=1
    )

    g.set(xlabel="Value of mu")

    plt.savefig(outpath / 'mu_distribution.svg')
    if show:
        plt.show()

    print('margin')
    print(stats.ks_2samp(top_mean_distances['margin'], bottom_mean_distances['margin']))
    print(stats.mannwhitneyu(top_mean_distances['margin'], bottom25_mean_distances['margin']))


    g = sns.displot(
        data=top_mean_distances['alpha'],
        binwidth=0.05,
        aspect=1
    )

    g.set(xlabel="Value of beta")

    plt.savefig(outpath / 'beta_distribution.svg')
    if show:
        plt.show()

    print('alpha')
    print(stats.ks_2samp(top_mean_distances['alpha'], mean_distances['alpha']))
    print(stats.mannwhitneyu(top_mean_distances['alpha'], mean_distances['alpha']))



    '''
    for column in top_mean_distances.columns:
        print(column)
        if column == 'alpha':
            binwidth=0.025
            binrange=-0.125
        elif column == 'adaptive_score_offset':
            binwidth=0.1
        elif column == 'gamma':
            binwidth=0.025
            binrange = -0.25, 0.325
        elif column == 'marigin':
            binwidth=0.00625
        else:
            binwidth=0.025

        print(binwidth)

        sns.displot(
            data=top_mean_distances[column],
            binwidth=binwidth,
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
    
    
    g = sns.scatterplot(
        data=mean_distances,
        x='gamma', y='margin',
        hue='mean_distance_to_orig',
        palette=cmap, alpha=0.2
    )

    plt.legend([], [], frameon=False)

    plt.show()
    '''

distances_distorted_can = pd.read_csv(Path('sets/hyperparameter/distances_distorted_can.csv'))
print(f"mean distance of distorted to original is: {np.mean(distances_distorted_can['distance_to_orig'])}")
for i in range(distances_distorted_can.shape[0]):
    distances_distorted_can.at[i, 'ssim_to_orig'] = 1 - distances_distorted_can.at[i, 'distance_to_orig']
distances_distorted_can.drop(columns=['distance_to_orig'], inplace=True)
print(f"mean ssim of distorted to original is: {np.mean(distances_distorted_can['ssim_to_orig'])}")

distances_ssiae = pd.read_csv(Path('sets/hyperparameter/hypersearch-0-28.csv'))
print(f"mean distance of H_opt to original is: {np.mean(distances_ssiae['distance_to_orig'])}")
for i in range(distances_ssiae.shape[0]):
    distances_ssiae.at[i, 'ssim_to_orig'] = 1 - distances_ssiae.at[i, 'distance_to_orig']
distances_ssiae.drop(columns=['distance_to_orig'], inplace=True)
print(f"mean ssim of H_opt to original is: {np.mean(distances_ssiae['ssim_to_orig'])}")

data = [distances_ssiae['image_id'], distances_distorted_can["ssim_to_orig"], distances_ssiae["ssim_to_orig"],
        (distances_ssiae["ssim_to_orig"] - distances_distorted_can["ssim_to_orig"])]
headers = ["image_id", "distance_orig_dist", "distance_orig_rec", "recovery_change"]

plot_ssim = pd.concat(data, axis=1, keys=headers)

html = plot_ssim.to_html()
with open(Path('results/distances_recovery_change_ssiae.html'), 'w') as file:
    file.write(html)

g = sns.scatterplot(
    data=plot_ssim,
    x='distance_orig_dist', y='recovery_change', s=20
)
plt.axhline(y=0.0, color='r', linestyle='-')
g.set(xlabel="SSIM between original and distorted image", ylabel="Change of SSIM after ours(SGD) was applied")
plt.ylim(-1, 1)

plt.savefig(outpath / 'ssim_recovery.svg')

if show:
    plt.show()

distances_nicer = pd.read_csv(Path('sets/hyperparameter/distances_nicer.csv'))
print(f"mean distance of NICER to original is: {np.mean(distances_nicer['distance_to_orig'])}")
for i in range(distances_nicer.shape[0]):
    distances_nicer.at[i, 'ssim_to_orig'] = 1 - distances_nicer.at[i, 'distance_to_orig']
distances_nicer.drop(columns=['distance_to_orig'], inplace=True)
print(f"mean ssim of NICER to original is: {np.mean(distances_nicer['ssim_to_orig'])}")

data = [distances_nicer['image_id'], distances_distorted_can["ssim_to_orig"], distances_nicer["ssim_to_orig"],
        (distances_nicer["ssim_to_orig"] - distances_distorted_can["ssim_to_orig"])]

headers = ["image_id", "distance_orig_dist", "distance_orig_rec", "recovery_change"]
plot_nicer = pd.concat(data, axis=1, keys=headers)

html = plot_nicer.to_html()
with open(Path('results/distances_recovery_change_nicer.html'), 'w') as file:
    file.write(html)

g = sns.scatterplot(
    data=plot_nicer,
    x='distance_orig_dist', y='recovery_change', s=20
)
plt.axhline(y=0.0, color='r', linestyle='-')
g.set(xlabel="SSIM between original and distorted image", ylabel="Change of SSIM after original NICER was applied")
plt.ylim(-1, 1)

plt.savefig(outpath / 'nicer.svg')

if show:
    plt.show()

distances_ssiae_cma = pd.read_csv(Path('sets/hyperparameter/hypersearch-10-46.csv'))
print(f"mean distance of H_opt_cma to original is: {np.mean(distances_ssiae_cma['distance_to_orig'])}")
for i in range(distances_ssiae_cma.shape[0]):
    distances_ssiae_cma.at[i, 'ssim_to_orig'] = 1 - distances_ssiae_cma.at[i, 'distance_to_orig']
distances_ssiae_cma.drop(columns=['distance_to_orig'], inplace=True)
print(f"mean ssim of H_opt_cma to original is: {np.mean(distances_ssiae_cma['ssim_to_orig'])}")

data = [distances_ssiae_cma['image_id'], distances_distorted_can["ssim_to_orig"], distances_ssiae_cma["ssim_to_orig"],
        (distances_ssiae_cma["ssim_to_orig"] - distances_distorted_can["ssim_to_orig"])]
headers = ["image_id", "distance_orig_dist", "distance_orig_rec", "recovery_change"]

plot_ssim_cma = pd.concat(data, axis=1, keys=headers)

g = sns.scatterplot(
    data=plot_ssim_cma,
    x='distance_orig_dist', y='recovery_change', s=20
)

plt.axhline(y=0.0, color='r', linestyle='-')
g.set(xlabel="SSIM between original and distorted image", ylabel="Change of SSIM after ours(CMA) was applied")
plt.ylim(-1, 1)

plt.savefig(outpath / 'ssim_cma_recovery.svg')
if show:
    plt.show()

print('overall test for significant improvement SGD')
print(stats.mannwhitneyu(plot_ssim['distance_orig_rec'], plot_ssim['distance_orig_dist'], alternative='greater'))

print('overall test for significant improvement CMA')
print(stats.mannwhitneyu(plot_ssim_cma['distance_orig_rec'], plot_ssim_cma['distance_orig_dist'], alternative='greater'))

print('overall ssia SGD vs nicer')
print(stats.mannwhitneyu(plot_ssim['recovery_change'], plot_nicer['recovery_change'], alternative='greater'))

print('overall ssia CMA vs nicer')
print(stats.mannwhitneyu(plot_ssim_cma['recovery_change'], plot_nicer['recovery_change'], alternative='greater'))

print('overall ssia CMA vs SGD')
print(stats.mannwhitneyu(plot_ssim['recovery_change'], plot_ssim_cma['recovery_change'], alternative='greater'))

print(f"mean ssim of H_opt_cma to original is: {np.mean(plot_ssim['recovery_change'])}")
print(f"mean ssim of H_opt_cma to original is: {np.mean(plot_ssim_cma['recovery_change'])}")

print('SSIM < 0.7 ssia vs nicer')

plot_ssim_s = plot_ssim[plot_ssim["distance_orig_dist"] < 0.7]
plot_nicer_s = plot_nicer[plot_nicer["distance_orig_dist"] < 0.7]

print(stats.mannwhitneyu(plot_ssim_s['recovery_change'], plot_nicer_s['recovery_change'], alternative='greater'))

print('0.7 test for significant improvement')
print(stats.mannwhitneyu(plot_ssim_s['distance_orig_rec'], plot_ssim_s['distance_orig_dist'], alternative='greater'))
print(stats.mannwhitneyu(plot_ssim_s['distance_orig_rec'], plot_ssim_s['distance_orig_dist'], alternative='greater')[1])

cols = []
for i in range(10, 5, -1):
    cutoff = i / 10
    cols.append(str(cutoff))

significance_tests = pd.DataFrame(columns=cols)

for i in range(5, 11):
    cutoff = i / 10
    if cutoff != 0.5:
        cutoff_low = cutoff - 0.1
    else:
        cutoff_low = 0

    plot_ssim_s = plot_ssim[plot_ssim["distance_orig_dist"] < cutoff]
    plot_ssim_s = plot_ssim_s[plot_ssim_s["distance_orig_dist"] > cutoff_low]

    plot_ssim_cma_s = plot_ssim_cma[plot_ssim_cma["distance_orig_dist"] < cutoff]
    plot_ssim_cma_s = plot_ssim_cma_s[plot_ssim_cma_s["distance_orig_dist"] > cutoff_low]

    plot_nicer_s = plot_nicer[plot_nicer["distance_orig_dist"] < cutoff]
    plot_nicer_s = plot_nicer_s[plot_nicer_s["distance_orig_dist"] > cutoff_low]

    significance_tests.at['amount_of_images', str(cutoff)] = plot_ssim_s.shape[0]
    significance_tests.at['mean_ssim', str(cutoff)] = "{:.3f}".format(np.mean(plot_ssim_s['distance_orig_dist']))

    significance_tests.at['mean_ssim_sgd', str(cutoff)] = "{:.3f}".format(np.mean(plot_ssim_s['distance_orig_rec']))
    significance_tests.at['mean_ssim_recovery_sgd', str(cutoff)] = "{:.3f}".format(np.mean(plot_ssim_s['recovery_change']))

    significance_tests.at['mean_ssim_cma', str(cutoff)] = "{:.3f}".format(np.mean(plot_ssim_cma_s['distance_orig_rec']))
    significance_tests.at['mean_ssim_recovery_cma', str(cutoff)] = "{:.3f}".format(np.mean(plot_ssim_cma_s['recovery_change']))

    significance_tests.at['mean_ssim_nicer', str(cutoff)] = "{:.3f}".format(np.mean(plot_nicer_s['distance_orig_rec']))
    significance_tests.at['mean_ssim_recovery_nicer', str(cutoff)] = "{:.3f}".format(np.mean(plot_nicer_s['recovery_change']))

    significance_tests.at['Significant_improvement', str(cutoff)] = "{:.2%}".format(stats.mannwhitneyu(plot_ssim_s['distance_orig_rec'], plot_ssim_s['distance_orig_dist'], alternative='greater')[1])
    significance_tests.at['Significant_improvement_cma', str(cutoff)] = "{:.2%}".format(stats.mannwhitneyu(plot_ssim_cma_s['distance_orig_rec'], plot_ssim_cma_s['distance_orig_dist'], alternative='greater')[1])

    significance_tests.at['Significantly_better_than_nicer', str(cutoff)] = "{:.2%}".format(stats.mannwhitneyu(plot_ssim_s['recovery_change'], plot_nicer_s['recovery_change'], alternative='greater')[1])
    significance_tests.at['Significantly_better_than_nicer_cma', str(cutoff)] = "{:.2%}".format(stats.mannwhitneyu(plot_ssim_cma_s['recovery_change'], plot_nicer_s['recovery_change'], alternative='greater')[1])

    significance_tests.at['Significantly_different_cma_sgd', str(cutoff)] = "{:.2%}".format(stats.mannwhitneyu(plot_ssim_cma_s['recovery_change'], plot_ssim_s['recovery_change'])[1])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(significance_tests)

print(significance_tests.to_latex())

g = sns.histplot(
    data=plot_ssim_cma,
    x='distance_orig_dist'
)
g.set(xlabel="SSIM between original and distorted image")

plt.savefig(outpath / 'ssim_dist.svg')
if show:
    plt.show()

print(sns.__version__)