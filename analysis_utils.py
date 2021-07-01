import pickle
import random
import json
import pandas as pd
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float

from dataset import Pexels, Pexels_hyperparamsearch
from utils import nima_transform, jans_transform, weighted_mean
from statistics import mean
from autobright import normalize_brightness
from PIL import Image
from pathlib import Path



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


def write_dict_to_file(dict, filename, path="./analysis/results/"):
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(path + filename + ".csv", sep=',', index=True)
    html = df.to_html()
    with open(path + filename + ".html", 'w') as file:
        file.write(html)


def add_to_dict(dict: dict, prefixes: list, postfix: str):
    for prefix in prefixes:
        dict[f'{prefix}{postfix}'] = []


def get_results_dict(prefixes: list, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine):
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
        nicer.forward(image_tensor_transformed, image_tensor_transformed_jan, headless_mode=True,
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
        print(f"processing {item['image_id']} in iteration {i}")
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

        # Evaluate unedited image and save scores to dictionary and a copy of the image to disk
        nicer.config.SSMTPIAA_loss = 'MSE_SCORE_REG'
        evaluate_image(item['img'], nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='orig')
        item['img'].save('./analysis/results/' + output_file + '/' + item['image_id'])

        for loss in losses:
            print(f"editing {item['image_id']} using {loss} in iteration {i}")

            # Set loss
            nicer.config.SSMTPIAA_loss = loss

            # Edit image
            edited_image, graph_data = nicer.enhance_image(item['img'], re_init=True, headless_mode=True,
                                                           nima_vgg16=nima_vgg16, nima_mobilenetv2=nima_mobilenetv2,
                                                           ssmtpiaa=ssmtpiaa, ssmtpiaa_fine=ssmtpiaa_fine)
            edited_image = Image.fromarray(edited_image)

            # Evaluate edited image and save scores
            evaluate_image(edited_image, nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix=loss)

            # Export image with rating history
            edited_image.save('./analysis/results/' + output_file + '/' + item['image_id'].split('.')[0] + loss + '.' + item['image_id'].split('.')[1])

            with open("./analysis/results/" + output_file + '/' + item['image_id'].split('.')[0] + loss + ".json", "w") as outfile:
                json.dump(graph_data, outfile)

        if i % 50 == 0:
            write_dict_to_file(results, output_file + str(i),
                               path="./analysis/results/" + output_file + '_graph_data' + "/")
            for key in results:
                results[key] = []

    if results['image_id']:
        write_dict_to_file(results, output_file + str(limit),
                           path="./analysis/results/" + output_file + '_graph_data' + "/")


def evaluate_editing_recovery_pexels(nicer, sample_size, img_path: Path, graph_data_path: Path, filename, loss: str, limit=None,
                            nima_vgg16=True, nima_mobilenetv2=True, ssmtpiaa=True, ssmtpiaa_fine=True):

    results = get_results_dict(['orig', 'dist', 'rest'], nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine)
    #results['dist_filters'] = []
    results['distance_to_orig'] = []

    pexels = Pexels_hyperparamsearch(sample_size=sample_size)

    if limit is None:
        limit = len(pexels)

    for i in range(limit):
        item = pexels.__getitem__(i)
        print('processing ' + str(item['image_id']) + ' in iteration ' + str(i))
        results['image_id'].append(item['image_id'])

        # Set loss
        nicer.config.SSMTPIAA_loss = loss


        # Evaluate unedited image and save scores to dictionary
        evaluate_image(item['img_orig'], nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='orig')

        img_orig = img_as_float(np.array(item['img_orig']))

        # Evaluate distorted image and save scores to dictionary
        evaluate_image(item['img_dist'], nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='dist')

        #img_dist = img_as_float(np.array(item['img_dist'])) #TODO probably not useful

        print(f"editing {item['image_id']} using {loss} in iteration {i}")

        # Edit image
        restored_image, graph_data = nicer.enhance_image(item['img_dist'], re_init=True, headless_mode=True,
                                                         nima_vgg16=nima_vgg16, nima_mobilenetv2=nima_mobilenetv2,
                                                         ssmtpiaa=ssmtpiaa, ssmtpiaa_fine=ssmtpiaa_fine)
        img_rest = img_as_float(np.array(restored_image))
        restored_image = Image.fromarray(restored_image)

        # Evaluate edited image and save scores
        evaluate_image(restored_image, nicer, results, nima_vgg16, nima_mobilenetv2, ssmtpiaa, ssmtpiaa_fine, prefix='rest')

        similarity = 1 - ssim(img_orig, img_rest, multichannel=True)
        results['distance_to_orig'].append(similarity)

        # Export image with rating history
        restored_image.save(img_path/f"{item['image_id'].split('.')[0]}_rest.{item['image_id'].split('.')[1]}")

        with open(img_path/f"{item['image_id'].split('.')[0]}_rest.json", "w") as outfile:
            json.dump(graph_data, outfile)

        if i+1 % 50 == 0:
            df = pd.DataFrame.from_dict(results)
            df.to_csv(graph_data_path / f"{filename}_{i}.csv", sep=',', index=True)
            html = df.to_html()
            with open(graph_data_path / f"{filename}_{i}.html", 'w') as file:
                file.write(html)
            # reset results dictionary
            for key in results:
                results[key] = []

    if results['image_id']:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(graph_data_path / f"_{limit}.csv", sep=',', index=True)
        html = df.to_html()
        with open(graph_data_path / f"_{limit}.html", 'w') as file:
            file.write(html)