import pandas as pd
import os
import random

from pathlib import Path

nicer = os.listdir(Path('datasets/div2ksurvey/nicer'))
original = os.listdir(Path('datasets/div2ksurvey/original'))
ssmtpiaa_cma = os.listdir(Path('datasets/div2ksurvey/ssmtpiaa_cma'))
ssmtpiaa_sgd = os.listdir(Path('datasets/div2ksurvey/ssmtpiaa_sgd'))
ssmtpiaa_sgd_oldloss = os.listdir(Path('datasets/div2ksurvey/ssmtpiaa_sgd_oldloss'))

debug = True
if debug:
    print(nicer)
    print(original)
    print(ssmtpiaa_cma)
    print(ssmtpiaa_sgd)
    print(ssmtpiaa_sgd_oldloss)

nicer_select = True
original_select = True
ssmtpiaa_cma_select = False
ssmtpiaa_sgd_select = True
ssmtpiaa_sgd_oldloss_select = True

selected = 0
if nicer_select:
    selected += 1
if original_select:
    selected += 1
if ssmtpiaa_cma_select:
    selected += 1
if ssmtpiaa_sgd_select:
    selected += 1
if ssmtpiaa_sgd_oldloss_select:
    selected += 1

print(selected)
styles = []
for i in range(selected):
    styles.append([])

for i in range(500):
    selection = []
    if nicer_select:
        selection.append(f"nicer/{nicer[i]}")
    if original_select:
        selection.append(f"original/{original[i]}")
    if ssmtpiaa_cma_select:
        selection.append(f"ssmtpiaa_cma/{ssmtpiaa_cma[i]}")
    if ssmtpiaa_sgd_select:
        selection.append(f"ssmtpiaa_sgd/{ssmtpiaa_sgd[i]}")
    if ssmtpiaa_sgd_oldloss_select:
        selection.append(f"ssmtpiaa_sgd_oldloss/{ssmtpiaa_sgd_oldloss[i]}")

    random.shuffle(selection)

    if debug:
        print(selection)

    for j in range(selected):
        styles[j].append(selection[j])

if debug:
    for style in styles:
        print(style)

styles_dict = {}
for i in range(selected):
    styles_dict[f"style_{i+1}"] = styles[i]

df = pd.DataFrame(styles_dict)
if debug:
    print(df)
df.to_csv(Path('analysis/sets/div2ksurvey_mturk'), index=True)