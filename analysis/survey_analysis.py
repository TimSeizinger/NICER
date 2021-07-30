import pandas as pd
import numpy as np

from pathlib import Path
from survey_objects import style_data

visualization = True

survey = pd.read_csv(Path('sets/survey_results/survey_results.csv'))

image_ids = list(set(survey['image_id'].tolist()))
image_ids.sort()

survey_grouped = pd.DataFrame({'image_id': image_ids})

styles = ['original', 'ssmtpiaa_cma']
best_styles = {}

for style in styles:
    best_styles[style] = 0

for i in range(len(image_ids)):
    ratings = survey[survey['image_id'] == image_ids[i]]
    for style in styles:
        image_ratings = ratings[f'{style}']
        survey_grouped.at[i, f'{style}_rating'] = np.mean(image_ratings)
        survey_grouped.at[i, f'{style}_var'] = np.var(image_ratings)
        survey_grouped.at[i, f'{style}_std'] = np.std(image_ratings)
        if visualization:
            survey_grouped.at[i, f'{style}_img'] = ratings[f'{style}_img'].tolist()[0]
            survey_grouped.at[i, style] = style_data(survey_grouped.at[i, f'{style}_rating'],
                                                        survey_grouped.at[i, f'{style}_var'],
                                                        survey_grouped.at[i, f'{style}_std'])

    '''
    best = 0

    for style in styles:
        best = max(best, survey_grouped.at[i, style])


    for style in styles:
         if survey_grouped.at[i, style] == best:
             best_styles[style] += 1
    '''

print(pd.DataFrame(survey_grouped.iloc[0]).to_html())

'''
for style in styles:
    print(f'Style {style} was the best in {best_styles[style]} cases')
'''

with open('sets/survey_results/processed/grouped_ratings.html', 'w') as file:
    html = survey_grouped.to_html(escape=False)
    file.write(html)

