import os
import shutil
import yaml
import pandas as pd

from pathlib import Path

in_dir = Path("out")

if not "batch_data.csv" in os.listdir(in_dir):
    yaml_dir = Path("k8s") / Path("hyperparametersearch")
    batches = os.listdir(yaml_dir)
    parameters = {'name': [], 'optim_lr': []}
    for batch in batches:
        yaml_files = os.listdir(yaml_dir / batch)
        print(f"Processing: {batch}")
        for yaml_file in yaml_files:
            parameters['name'].append(yaml_file.split('.')[0])
            with open(yaml_dir / batch / yaml_file, 'r') as stream:
                try:
                    yaml_data = yaml.safe_load(stream)
                    optim_lr = f"_{yaml_data['spec']['template']['spec']['containers'][0]['command'][3]}"
                    parameters['optim_lr'].append(optim_lr)
                except yaml.YAMLError as exc:
                    print(exc)

    batch_data = pd.DataFrame(parameters)
    print(batch_data)

    batch_data.to_csv(in_dir / "batch_data.csv")
else:
    print("loading batch_data.csv")
    batch_data = pd.read_csv(in_dir / "batch_data.csv")

subfolders = os.listdir(in_dir)

candidates_1500 = []
candidates_3000 = []
candidates_3000_length = []
mean_distances = []
yaml_names = []
parameters = []

for subfolder in subfolders:
    print(f"Processing: {subfolder}")
    if (subfolder == "sub") | (subfolder == "successful") | (subfolder == "test") | (subfolder == "batch_data.csv") | (subfolder == "suggestions.csv") | (subfolder == "mean_distances.csv"):
        continue

    checkpoints = os.listdir(in_dir / subfolder / "data")
    if not checkpoints:
        print(f"Empty data directory")
        continue

    optim_lr = f"_{checkpoints[0].split('_')[0]}"

    if optim_lr == "_0.02":
        yaml_name = "handpicked"
    else:
        yaml_name = batch_data.loc[batch_data['optim_lr'] == optim_lr].iloc[0][1]
    print(f"Yaml: {yaml_name}")

    checkpoint_iterations = [int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in checkpoints]

    maximum_checkpoint = max(checkpoint_iterations)
    print(f"{yaml_name} has max checkpoint of {maximum_checkpoint}")

    if maximum_checkpoint == 4999:

        checkpoint_data = []
        for checkpoint in checkpoints:
            checkpoint_data.append(pd.read_csv(in_dir / subfolder / "data" / checkpoint))

        combined_data = pd.concat(checkpoint_data, ignore_index=True)
        combined_data.drop(columns='Unnamed: 0', inplace=True)
        mean_distance = combined_data['distance_to_orig'].mean()

        print(f"mean distances_to_orig of {yaml_name} is: {mean_distance}")

        yaml_names.append(yaml_name)
        mean_distances.append(mean_distance)
        parameters.append(subfolder)

        results = {'yaml_name': yaml_names, 'mean_distance_to_orig': mean_distances, 'parameters': parameters}
        results_df = pd.DataFrame(results)
        results_df.to_csv(in_dir / 'successful' / "mean_distances.csv")
        results_df.sort_values(by='mean_distance_to_orig', ascending=True, inplace=True)
        results_df.to_csv(in_dir / 'successful' / "mean_distances_sorted.csv")
    elif maximum_checkpoint > 3000:
        candidates_3000.append(batch_data.loc[batch_data['optim_lr'] == optim_lr].iloc[0][1])
        candidates_3000_length.append(maximum_checkpoint)
        print("Added to 3000+ candidates")
    elif maximum_checkpoint > 1500:
        candidates_1500.append(batch_data.loc[batch_data['optim_lr'] == optim_lr].iloc[0][1])
        print("Added to 1500+ candidates")


candidates = {'3000': candidates_3000, 'length': candidates_3000_length}
candidates_df = pd.DataFrame(candidates)
candidates_df.to_csv(in_dir / "suggestions.csv")

results = {'yaml_name': yaml_names, 'mean_distance_to_orig': mean_distances, 'parameters': parameters}
results_df = pd.DataFrame(results)
results_df.to_csv(in_dir / 'successful' / "mean_distances.csv")
results_df.sort_values(by='mean_distance_to_orig', ascending=True, inplace=True)
results_df.to_csv(in_dir / 'successful' / "mean_distances_sorted.csv")