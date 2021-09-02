import pandas as pd
import numpy as np
import random
import math

from survey_objects import style_data
from PIL import Image, ImageStat

def combine_dataframes(df1 :pd.DataFrame, df2 :pd.DataFrame):
    combined = pd.concat([df1, df2])

    combined = combined[combined.Approve == 'x']
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    combined.drop('Unnamed: 0', inplace=True, axis=1)

    return combined

def get_ordered_results(df :pd.DataFrame, visualization :bool):

    survey_result = pd.DataFrame(
        {'worker': df['WorkerId'], 'duration': df['WorkTimeInSeconds']})

    for i in range(len(survey_result['duration'])):

        survey_result.at[i, 'image_id'] = int(
            df[f"Input.style_1"][i].split('/')[1].split('_')[0].split('-')[1])

        for j in range(1, 6):
            if df[f'Input.style_{j}'][i].split('/')[0] == 'original':
                survey_result.at[i, 'original_rating'] = df[f'Answer.rating_{j}'][i]
                survey_result.at[i, 'original_img_link'] = df[f"Input.style_{j}"][i]
                if visualization:
                    survey_result.at[i, 'original_img'] = \
                        f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{df[f"Input.style_{j}"][i]}" ' \
                        f'style="max-width: 200px; max-height: 200px" loading="lazy">'
            elif df[f'Input.style_{j}'][i].split('/')[0] == 'nicer':
                survey_result.at[i, 'nicer_rating'] = df[f'Answer.rating_{j}'][i]
                survey_result.at[i, 'nicer_img_link'] = df[f"Input.style_{j}"][i]
                if visualization:
                    survey_result.at[i, 'nicer_img'] = \
                        f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{df[f"Input.style_{j}"][i]}" ' \
                        f'style="max-width: 200px; max-height: 200px" loading="lazy">'
            elif df[f'Input.style_{j}'][i].split('/')[0] == 'ssmtpiaa_sgd':
                survey_result.at[i, 'ssmtpiaa_sgd_rating'] = df[f'Answer.rating_{j}'][i]
                survey_result.at[i, 'ssmtpiaa_sgd_img_link'] = df[f"Input.style_{j}"][i]
                if visualization:
                    survey_result.at[i, 'ssmtpiaa_sgd_img'] = \
                        f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{df[f"Input.style_{j}"][i]}" ' \
                        f'style="max-width: 200px; max-height: 200px" loading="lazy">'
            elif df[f'Input.style_{j}'][i].split('/')[0] == 'ssmtpiaa_cma':
                survey_result.at[i, 'ssmtpiaa_cma_rating'] = df[f'Answer.rating_{j}'][i]
                survey_result.at[i, 'ssmtpiaa_cma_img_link'] = df[f"Input.style_{j}"][i]
                if visualization:
                    survey_result.at[i, 'ssmtpiaa_cma_img'] = \
                        f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{df[f"Input.style_{j}"][i]}" ' \
                        f'style="max-width: 200px; max-height: 200px" loading="lazy">'
            elif df[f'Input.style_{j}'][i].split('/')[0] == 'expert':
                survey_result.at[i, 'expert_rating'] = df[f'Answer.rating_{j}'][i]
                survey_result.at[i, 'expert_img_link'] = df[f"Input.style_{j}"][i]
                if visualization:
                    survey_result.at[
                        i, 'expert_img'] = f'<img src="https://vps.pfstr.de/f/mturk/adobe5k/{df[f"Input.style_{j}"][i]}" style="max-width: 200px; max-height: 200px" loading="lazy">'

    survey_result.image_id = survey_result.image_id.astype(int)
    survey_result.duration = survey_result.duration.astype(int)

    survey_result.sort_values(by=['image_id'], inplace=True)

    survey_result.reset_index(inplace=True)
    survey_result.drop('index', inplace=True, axis=1)

    return survey_result

def preprocess_data(df1 :pd.DataFrame, df2 :pd.DataFrame, visualization :bool):
    return get_ordered_results(combine_dataframes(df1, df2), visualization)

def combine_image_ratings(survey :pd.DataFrame, visualization :bool,
                          styles :list = ['original', 'nicer', 'ssmtpiaa_sgd', 'ssmtpiaa_cma', 'expert']):
    image_ids = list(set(survey['image_id'].tolist()))
    image_ids.sort()

    survey_grouped = pd.DataFrame({'image_id': image_ids})

    for i in range(len(image_ids)):
        ratings = survey[survey['image_id'] == image_ids[i]]
        for style in styles:
            image_ratings = ratings[f'{style}_rating']
            survey_grouped.at[i, f'{style}_rating'] = np.mean(image_ratings)
            survey_grouped.at[i, f'{style}_var'] = np.var(image_ratings)
            survey_grouped.at[i, f'{style}_std'] = np.std(image_ratings)
            if visualization:
                survey_grouped.at[i, f'{style}_img'] = ratings[f'{style}_img'].tolist()[0]
                survey_grouped.at[i, style] = style_data(survey_grouped.at[i, f'{style}_rating'],
                                                         survey_grouped.at[i, f'{style}_var'],
                                                         survey_grouped.at[i, f'{style}_std'])

    return survey_grouped

def get_relevant_columns_for_visualization(visualization):
    if visualization:
        return ['image_id', 'original', 'original_img', 'nicer', 'nicer_img', 'ssmtpiaa_sgd', 'ssmtpiaa_sgd_img', 'ssmtpiaa_cma', 'ssmtpiaa_cma_img', 'expert', 'expert_img']

def get_random_array(length):
    random_list = []
    for i in range(length):
        random_list.append(random.randint(0, 1))
    return random_list

def get_brightness(im_path, read=True):
    if read:
        im_file = Image.open(im_path)
    else:
        im_file = im_path  # passed "path" already is PIL image
    stat = ImageStat.Stat(im_file)
    try:
        r, g, b = stat.mean  # RGB
        res = math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
    except ValueError:
        mean = stat.mean  # grayscale
        res = math.sqrt(mean[0])
    return res

def check_if_unambiguous(survey_result, candidate, styles, i, best):
    for style in styles:
        if style == candidate:
            continue
        if survey_result.at[i, f'{style}_rating'] == best:
            return False
    return True

def count_best(survey_result, styles_to_evaluate, visualization):
    # initialize counter
    best_styles = {}
    examples = {}

    for style in styles_to_evaluate:
        best_styles[style] = 0
        examples[style] = set()
    even_correction = 0

    for i in range(survey_result.shape[0]):
        best = 0
        even_flag = False
        # Get best rating for each image
        for style in styles_to_evaluate:
            best = max(best, survey_result.at[i, f'{style}_rating'])
        # Find out to which style it belongs
        for style in styles_to_evaluate:
            if survey_result.at[i, f'{style}_rating'] == best:
                if check_if_unambiguous(survey_result, style, styles_to_evaluate, i, best):
                    best_styles[style] += 1
                    examples[style].add(i)
                else:
                    even_flag = True
        if even_flag:
            even_correction += 1

    if visualization:
        return best_styles, examples
    else:
        return best_styles
