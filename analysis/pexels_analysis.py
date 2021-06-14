import pandas as pd
import os

os.chdir(os.path.dirname(os.getcwd()))

pexels_tall = pd.read_csv('./analysis/results/pexels_tall.csv')
pexels_wide = pd.read_csv('./analysis/results/pexels_wide.csv')


print('pexels_tall orig mean SSMTPIAA score: ' + str(pexels_tall['orig_ia_pre_score'].mean()))
print('pexels_wide orig mean SSMTPIAA score: ' + str(pexels_wide['orig_ia_pre_score'].mean()))

'''
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
'''
