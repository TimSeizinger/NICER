import pandas as pd
import numpy as np

from dateutil.parser import parse
from pathlib import Path

survey = pd.read_csv(Path('sets/survey_results/MTurk_Batch_2.csv'))

analysed = pd.DataFrame(survey['WorkerId'])
analysed['duration'] = np.nan
analysed['original'] = np.nan
analysed['original_img'] = np.nan
analysed['nicer'] = np.nan
analysed['nicer_img'] = np.nan
analysed['ssmtpiaa_sgd'] = np.nan
analysed['ssmtpiaa_sgd_img'] = np.nan
analysed['ssmtpiaa_cma'] = np.nan
analysed['ssmtpiaa_cma_img'] = np.nan
analysed['expert'] = np.nan
analysed['expert_img'] = np.nan

analysed['rejected'] = np.nan


for i in range(len(analysed['duration'])):
    start = parse(survey['AcceptTime'][i])
    end = parse(survey['SubmitTime'][i])
    analysed['duration'][i] = (end - start).total_seconds()

    for j in range(1,6):
        if survey[f'Input.style_{j}'][i].split('/')[0] == 'original':
            analysed['original'][i] = survey[f'Answer.rating_{j}'][i]
            analysed['original_img'][i] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey[f'Input.style_{j}'][i].split('/')[0] == 'nicer':
            analysed['nicer'][i] = survey[f'Answer.rating_{j}'][i]
            analysed['nicer_img'][
                i] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey[f'Input.style_{j}'][i].split('/')[0] == 'ssmtpiaa_sgd':
            analysed['ssmtpiaa_sgd'][i] = survey[f'Answer.rating_{j}'][i]
            analysed['ssmtpiaa_sgd_img'][
                i] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey[f'Input.style_{j}'][i].split('/')[0] == 'ssmtpiaa_cma':
            analysed['ssmtpiaa_cma'][i] = survey[f'Answer.rating_{j}'][i]
            analysed['ssmtpiaa_cma_img'][
                i] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'
        elif survey[f'Input.style_{j}'][i].split('/')[0] == 'expert':
            analysed['expert'][i] = survey[f'Answer.rating_{j}'][i]
            analysed['expert_img'][
                i] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{survey[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'

    defaults = 0
    if analysed['original'][i] == 5.5:
        defaults += 1
    if analysed['nicer'][i] == 5.5:
        defaults += 1
    if analysed['ssmtpiaa_sgd'][i] == 5.5:
        defaults += 1
    if analysed['ssmtpiaa_cma'][i] == 5.5:
        defaults += 1
    if analysed['expert'][i] == 5.5:
        defaults += 1

    if defaults >= 3:
        analysed['rejected'][i] = True
        survey['Reject'][i] = 'More than two default values'
    else:
        answerset = {analysed['original'][i], analysed['nicer'][i], analysed['ssmtpiaa_sgd'][i], analysed['ssmtpiaa_cma'][i], analysed['expert'][i]}
        if len(answerset) <= 3:
            analysed['rejected'][i] = True
            survey['Reject'][i] = f'You only provided {len(answerset)} distinct ratings. You were supposed to rate most images differently.'
        else:
            survey['Approve'][i] = 'x'



analysed.to_csv(Path('sets/survey_results/processed/MTurk_Batch_2_analysed.csv'))
with open('sets/survey_results/processed/MTurk_Batch_2_analysed.html', 'w') as file:
    html = analysed.to_html(escape=False)
    file.write(html)

survey.to_csv(Path('sets/survey_results/processed/MTurk_Batch_2_approvals.csv'))
with open('sets/survey_results/processed/MTurk_Batch_2_approvals.html', 'w') as file:
    html = survey.to_html(escape=False)
    file.write(html)
