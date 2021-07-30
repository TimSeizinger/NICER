import pandas as pd
import numpy as np

from dateutil.parser import parse
from pathlib import Path

visualization = True  # Include a html with embedded images.

survey1 = pd.read_csv(Path('sets/survey_results/processed/MTurk_Batch_1_approvals.csv'))
survey2 = pd.read_csv(Path('sets/survey_results/processed/MTurk_Batch_2_approvals.csv'))

survey_combined = pd.concat([survey1, survey2])

survey_combined = survey_combined[survey_combined.Approve == 'x']
survey_combined.reset_index(inplace=True)
survey_combined.drop('index', inplace=True, axis=1)
survey_combined.drop('Unnamed: 0', inplace=True, axis=1)


survey_result = pd.DataFrame({'worker': survey_combined['WorkerId'], 'duration': survey_combined['WorkTimeInSeconds']})

for i in range(len(survey_result['duration'])):

    survey_result.at[i, 'image_id'] = int(survey_combined[f"Input.style_1"][i].split('/')[1].split('_')[0].split('-')[1])

    for j in range(1,6):
        if survey_combined[f'Input.style_{j}'][i].split('/')[0] == 'original':
            survey_result.at[i, 'original'] = survey_combined[f'Answer.rating_{j}'][i]
            survey_result.at[i, 'original_img_link'] = survey_combined[f"Input.style_{j}"][i]
            if visualization:
                survey_result.at[i, 'original_img'] = \
                    f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey_combined[f"Input.style_{j}"][i]}" ' \
                    f'style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey_combined[f'Input.style_{j}'][i].split('/')[0] == 'nicer':
            survey_result.at[i, 'nicer'] = survey_combined[f'Answer.rating_{j}'][i]
            survey_result.at[i, 'nicer_img_link'] = survey_combined[f"Input.style_{j}"][i]
            if visualization:
                survey_result.at[i, 'nicer_img'] = \
                    f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey_combined[f"Input.style_{j}"][i]}" ' \
                    f'style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey_combined[f'Input.style_{j}'][i].split('/')[0] == 'ssmtpiaa_sgd':
            survey_result.at[i, 'ssmtpiaa_sgd'] = survey_combined[f'Answer.rating_{j}'][i]
            survey_result.at[i, 'ssmtpiaa_sgd_img_link'] = survey_combined[f"Input.style_{j}"][i]
            if visualization:
                survey_result.at[i, 'ssmtpiaa_sgd_img'] = \
                    f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey_combined[f"Input.style_{j}"][i]}" ' \
                    f'style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey_combined[f'Input.style_{j}'][i].split('/')[0] == 'ssmtpiaa_cma':
            survey_result.at[i, 'ssmtpiaa_cma'] = survey_combined[f'Answer.rating_{j}'][i]
            survey_result.at[i, 'ssmtpiaa_cma_img_link'] = survey_combined[f"Input.style_{j}"][i]
            if visualization:
                survey_result.at[i, 'ssmtpiaa_cma_img'] = \
                    f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey_combined[f"Input.style_{j}"][i]}" ' \
                f'style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey_combined[f'Input.style_{j}'][i].split('/')[0] == 'expert':
            survey_result.at[i, 'expert'] = survey_combined[f'Answer.rating_{j}'][i]
            survey_result.at[i, 'expert_img_link'] = survey_combined[f"Input.style_{j}"][i]
            if visualization:
                survey_result.at[i, 'expert_img'] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey_combined[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'

survey_result.image_id = survey_result.image_id.astype(int)
survey_result.duration = survey_result.duration.astype(int)

survey_result.sort_values(by=['image_id'], inplace=True)

survey_result.reset_index(inplace=True)
survey_result.drop('index', inplace=True, axis=1)

print(f"Mean of original: {np.mean(survey_result['original'])}")
print(f"Mean of nicer: {np.mean(survey_result['nicer'])}")
print(f"Mean of ssmtpiaa_sgd: {np.mean(survey_result['ssmtpiaa_sgd'])}")
print(f"Mean of ssmtpiaa_cma: {np.mean(survey_result['ssmtpiaa_cma'])}")
print(f"Mean of expert: {np.mean(survey_result['expert'])}")

survey_result.to_csv(Path('sets/survey_results/survey_results.csv'), index=False)