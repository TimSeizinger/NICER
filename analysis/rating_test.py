import config
import os
import random
import pandas as pd

from dataset import AVA
from utils import nima_transform, jans_normalization, weighted_mean
from nicer import NICER

pwd = os.getcwd()
os.chdir(os.path.dirname(os.getcwd()))

results = {'image_id': [], 'ava_score': [],
           'orig_nima_vgg16_score': [], 'orig_nima_mobilenetv2_score': [], 'orig_ia_fine_score': [], 'orig_ia_pre_score': [],
           'orig_ia_pre_styles_change': [],
           #'orig_ia_pre_styles_1': [], 'orig_ia_pre_styles_2': [], 'orig_ia_pre_styles_3': [],
           #'orig_ia_pre_styles_4': [], 'orig_ia_pre_styles_5': [], 'orig_ia_pre_styles_6': [],
           'dist_nima_vgg16_score': [], 'dist_nima_mobilenetv2_score': [], 'dist_ia_fine_score': [], 'dist_ia_pre_score': [],
           #'dist_ia_pre_styles_1': [], 'dist_ia_pre_styles_2': [], 'dist_ia_pre_styles_3': [],
           #'dist_ia_pre_styles_4': [], 'dist_ia_pre_styles_5': [], 'dist_ia_pre_styles_6': []
           'dist_ia_pre_styles_change': [], 'dist_filters': []
           }

nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path)
ava = AVA(mode='rating_test')

fixFilters = [1, 1, 1, 1, 1, 1, 1, 1]
filters = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for i in range(100):
    item = ava.__getitem__(i)

    print('processing ' + str(item['image_id']) + ' in iteration ' + str(i))
    results['image_id'].append(item['image_id'])
    results['ava_score'].append(item['ava_score'])

    pil_img = item['img']
    #bright_normalized_img = normalize_brightness(pil_img, input_is_PIL=True)
    #pil_img = Image.fromarray(bright_normalized_img)
    image_tensor_transformed = nima_transform(pil_img)
    image_tensor_jan = jans_normalization(pil_img)

    nicer.re_init()
    initial_filter_values = []
    for k in range(8):
        if fixFilters[k] == 1:
            initial_filter_values.append([1, nicer.filters[k].item()])
        else:
            initial_filter_values.append([0, nicer.filters[k].item()])

    unused, nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, blub, ia_fine_distr_of_ratings = nicer.forward(image_tensor_transformed)
    asfas, sdd, safas, ia_pre_ratings, fasfa = nicer.forward(
        image_tensor_jan)

    results['orig_nima_vgg16_score'].append(weighted_mean(nima_vgg16_distr_of_ratings, nicer.weights, nicer.length).item()*10)
    results['orig_nima_mobilenetv2_score'].append(weighted_mean(nima_mobilenetv2_distr_of_ratings, nicer.weights, nicer.length).item()*10)
    results['orig_ia_fine_score'].append(weighted_mean(ia_fine_distr_of_ratings, nicer.weights, nicer.length).item()*10)
    results['orig_ia_pre_score'].append(ia_pre_ratings['score'].item())
    results['orig_ia_pre_styles_change'].append(ia_pre_ratings['styles_change_strength'].squeeze().tolist())

    #Distorted image
    min, max = -1, 1
    filters = [random.uniform(min, max), random.uniform(min, max), random.uniform(min, max), random.uniform(min, max),
               random.uniform(min, max), random.uniform(min, max), random.uniform(min, max), random.uniform(min, max)]

    results['dist_filters'].append(filters)

    nicer.re_init()
    initial_filter_values = []
    nicer.set_filters(filters)
    for k in range(8):
        if fixFilters[k] == 1:
            initial_filter_values.append([1, nicer.filters[k].item()])
        else:
            initial_filter_values.append([0, nicer.filters[k].item()])

    unused, nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, _, ia_fine_distr_of_ratings = nicer.forward(image_tensor_transformed)
    unused, _, _, ia_pre_ratings, _ = nicer.forward(
        image_tensor_jan)

    results['dist_nima_vgg16_score'].append(weighted_mean(nima_vgg16_distr_of_ratings, nicer.weights, nicer.length).item()*10)
    results['dist_nima_mobilenetv2_score'].append(weighted_mean(nima_mobilenetv2_distr_of_ratings, nicer.weights, nicer.length).item()*10)
    results['dist_ia_fine_score'].append(weighted_mean(ia_fine_distr_of_ratings, nicer.weights, nicer.length).item()*10)
    results['dist_ia_pre_score'].append(ia_pre_ratings['score'].item())
    results['dist_ia_pre_styles_change'].append(ia_pre_ratings['styles_change_strength'].squeeze().tolist())

df = pd.DataFrame.from_dict(results)
df.to_csv("./analysis/results_bottom.csv", sep=',', index=True)
html = df.to_html()
with open('./analysis/results_bottom.html', 'w') as file:
    file.write(html)