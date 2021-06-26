import pickle
import json
import pandas as pd

from dataset import Pexels
from utils import nima_transform, jans_transform, weighted_mean
from statistics import mean
from autobright import normalize_brightness
from PIL import Image



def abs_mean(list):
    out = []
    for i in list:
        out.append(abs(i))
    return mean(out)


def preprocess_image(pil_img, only_jans_transform=False):
    print(type(pil_img))
    pil_img = Image.fromarray(normalize_brightness(pil_img, input_is_PIL=True))
    if only_jans_transform:
        return jans_transform(pil_img)
    return nima_transform(pil_img), jans_transform(pil_img)


def get_initial_fixed_filters(nicer):
    fixFilters = [1, 1, 1, 1, 1, 1, 1, 1]
    initial_filter_values = []
    for k in range(8):
        if fixFilters[k] == 1:
            initial_filter_values.append([1, nicer.filters[k].item()])
        else:
            initial_filter_values.append([0, nicer.filters[k].item()])
    return initial_filter_values

def write_dict_to_file(dict, filename):
    df = pd.DataFrame.from_dict(dict)
    df.to_csv("./analysis/results/" + filename + ".csv", sep=',', index=True)
    html = df.to_html()
    with open("./analysis/results/" + filename + ".html", 'w') as file:
        file.write(html)

def add_to_dict(dict: dict, prefixes: list, postfix: str):
    for prefix in prefixes:
        dict[f'{prefix}{postfix}'] = []

def get_results_dict(prefixes, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine):
    results = {'image_id': []}

    if nima_vgg16:
        add_to_dict(results, prefixes, '_nima_vgg16_score')
    if nima_mobilenetv2:
        add_to_dict(results, prefixes, '_nima_mobilenetv2_score')
    if ssmtpiaa:
        add_to_dict(results, prefixes, '_ia_pre_score')
        add_to_dict(results, prefixes, '_ia_pre_styles_change')
        add_to_dict(results, prefixes, '_ia_pre_styles_change_strength')
    if ssmtpiaa_fine:
        add_to_dict(results, prefixes, '_ia_fine_score')
    return results


def evaluate_image(image: Image, nicer, results, nima_vgg16=True, nima_mobilenetv2=True, ssmtpiaa=True, ssmtpiaa_fine=True, prefix='orig'):
    # Get scores for image
    if nima_vgg16:
        image_tensor_transformed, image_tensor_transformed_jan = \
            preprocess_image(image)
    else:
        image_tensor_transformed_jan = preprocess_image(image, only_jans_transform=True)
        image_tensor_transformed = None

    nicer.re_init()

    _, nima_vgg16_distr_of_ratings, nima_mobilenetv2_distr_of_ratings, ia_pre_ratings, ia_fine_distr_of_ratings = \
        nicer.forward(image_tensor_transformed, image_tensor_transformed_jan,
                      nima_vgg16=nima_vgg16, nima_mobilenetv2=nima_mobilenetv2,
                      ssmtpiaa=ssmtpiaa, ssmtpiaa_fine=ssmtpiaa_fine)

    # Write scores to results dict
    if nima_vgg16:
        results[f'{prefix}_nima_vgg16_score'].append(
            weighted_mean(nima_vgg16_distr_of_ratings, nicer.weights, nicer.length).item() * 10)
    if nima_mobilenetv2:
        results[f'{prefix}_nima_mobilenetv2_score'].append(
            weighted_mean(nima_mobilenetv2_distr_of_ratings, nicer.weights, nicer.length).item() * 10)
    if ssmtpiaa:
        results[f'{prefix}_ia_pre_score'].append(ia_pre_ratings['score'].item())
        results[f'{prefix}_ia_pre_styles_change'].append(ia_pre_ratings['styles_change_strength'].squeeze().tolist())
        results[f'{prefix}_ia_pre_styles_change_strength'].append(
            abs_mean(ia_pre_ratings['styles_change_strength'].squeeze().tolist()))
    if ssmtpiaa_fine:
        results[f'{prefix}_ia_fine_score'].append(
            weighted_mean(ia_fine_distr_of_ratings, nicer.weights, nicer.length).item() * 10)

def evaluate_rating_pexels(nicer, output_file, mode, limit=None):
    results = {'image_id': [],
               'orig_nima_vgg16_score': [], 'orig_nima_mobilenetv2_score': [], 'orig_ia_fine_score': [],
               'orig_ia_pre_score': [],
               'orig_ia_pre_styles_change': [], 'orig_ia_pre_styles_change_strength': [],
               }



    pexels = Pexels(mode=mode)

    if limit is None:
        limit = len(pexels)

    for i in range(limit):
        item = pexels.__getitem__(i)

        print('processing ' + str(item['image_id']) + ' in iteration ' + str(i))
        results['image_id'].append(item['image_id'])

        evaluate_image(item['img'], nicer, results, prefix='orig')

    write_dict_to_file(results, output_file)


def evaluate_editing_pexels(nicer, output_file, mode, limit=None,
                            nima_vgg16=True, nima_mobilenetv2=True, ssmtpiaa=True, ssmtpiaa_fine=True):

    results = get_results_dict(['orig', 'edit'], nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine)

    pexels = Pexels(mode=mode)

    if limit is None:
        limit = len(pexels)

    for i in range(limit):
        item = pexels.__getitem__(i)
        print('processing ' + str(item['image_id']) + ' in iteration ' + str(i))
        results['image_id'].append(item['image_id'])

        # Evaluate unedited image and save scores
        evaluate_image(item['img'], nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='orig')

        # Edit image
        edited_image, graph_data = nicer.enhance_image(item['img'], re_init=True, headless_mode=True,
                                                       nima_vgg16=nima_vgg16, nima_mobilenetv2=nima_mobilenetv2,
                                                       ssmtpiaa=ssmtpiaa, ssmtpiaa_fine=ssmtpiaa_fine)
        edited_image = Image.fromarray(edited_image)

        # Evaluate edited image and save scores
        evaluate_image(edited_image, nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='edit')

        # Export image with rating history
        item['img'].save('./analysis/results/' + output_file + '/' + item['image_id'])
        edited_image.save('./analysis/results/' + output_file + '/' + item['image_id'].split('.')[0] + '_edited' + '.' + item['image_id'].split('.')[1])

        with open("./analysis/results/" + output_file + '/' + item['image_id'].split('.')[0] + ".json", "w") as outfile:
            json.dump(graph_data, outfile)

    write_dict_to_file(results, output_file)

def evaluate_editing_losses_pexels(nicer, output_file, mode, losses: list, limit=None,
                            nima_vgg16=True, nima_mobilenetv2=True, ssmtpiaa=True, ssmtpiaa_fine=True):

    results = get_results_dict(['orig'] + losses, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine)

    pexels = Pexels(mode=mode)

    if limit is None:
        limit = len(pexels)

    for i in range(limit):
        item = pexels.__getitem__(i)
        print('processing ' + str(item['image_id']) + ' in iteration ' + str(i))
        results['image_id'].append(item['image_id'])

        # Evaluate unedited image and save scores
        evaluate_image(item['img'], nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='orig')

        for loss in losses:
            # Set loss
            nicer.config.SSMTPIAA_loss = loss

            # Edit image
            edited_image, graph_data = nicer.enhance_image(item['img'], re_init=True, headless_mode=True)
            edited_image = Image.fromarray(edited_image)

            # Evaluate edited image and save scores
            evaluate_image(edited_image, nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='edit')

            # Export image with rating history
            item['img'].save('./analysis/results/' + output_file + '/' + item['image_id'])
            edited_image.save('./analysis/results/' + output_file + '/' + item['image_id'].split('.')[0] + '_edited' + '.' + item['image_id'].split('.')[1])

            with open("./analysis/results/" + output_file + '/' + item['image_id'].split('.')[0] + ".json", "w") as outfile:
                json.dump(graph_data, outfile)

    write_dict_to_file(results, output_file)
