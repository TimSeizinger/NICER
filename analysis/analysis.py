import pandas as pd

ava_data = pd.read_csv('results.csv')
ava_ia_pre_data = ava_data[['orig_ia_pre_score', 'orig_ia_pre_styles_change', 'dist_ia_pre_score', 'dist_ia_pre_styles_change', 'dist_filters']]
ava_bottom_data = pd.read_csv('results_bottom.csv')

landscapes_data = pd.read_csv('results_landscapes.csv')
landscapes_ia_pre_data = landscapes_data[['orig_ia_pre_score', 'orig_ia_pre_styles_change', 'dist_ia_pre_score', 'dist_ia_pre_styles_change', 'dist_filters']]

print('AVA top 1000 orig mean ia_pre score: ' + str(ava_ia_pre_data['orig_ia_pre_score'].mean()))
print('AVA top 1000 orig variance ia_pre score: ' + str(ava_ia_pre_data['orig_ia_pre_score'].var()))
print('AVA top 1000 dist mean ia_pre score: ' + str(ava_ia_pre_data['dist_ia_pre_score'].mean()))
print('AVA top 1000 distorted variance ia_pre score: ' + str(ava_ia_pre_data['dist_ia_pre_score'].var()))

print('AVA bottom 100 orig mean ia_pre score: ' + str(ava_bottom_data['orig_ia_pre_score'].mean()))
print('AVA bottom 100 orig var ia_pre score: ' + str(ava_bottom_data['orig_ia_pre_score'].var()))

print('Landscapes orig mean score: ' + str(landscapes_ia_pre_data['orig_ia_pre_score'].mean()))
print('Landscapes orig var score: ' + str(landscapes_ia_pre_data['orig_ia_pre_score'].var()))
#print(ava_ia_pre_data['orig_ia_pre_styles_change'].values)

#print('AVA dist mean score: ' + str(ava_ia_pre_data['dist_ia_pre_score'].mean()))
print('Landscapes dist mean score: ' + str(landscapes_ia_pre_data['dist_ia_pre_score'].mean()))
print('Landscapes dist var score: ' + str(landscapes_ia_pre_data['dist_ia_pre_score'].var()))
