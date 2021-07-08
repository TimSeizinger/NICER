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

out_dir = Path("out") / Path("successful")


if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

subfolders = os.listdir(in_dir)

candidates_1500 = []
candidates_3000 = []

for subfolder in subfolders:
    print(f"Processing: {subfolder}")
    if (subfolder == "sub") | (subfolder == "successful") | (subfolder == "test") | (subfolder == "batch_data.csv") | (subfolder == "suggestions.csv"):
        continue

    subfolder = Path(subfolder)

    checkpoints = os.listdir(in_dir / subfolder / "data")
    if not checkpoints:
        print(f"Empty data directory")
        continue

    optim_lr = f"_{checkpoints[0].split('_')[0]}"

    yaml_name = batch_data.loc[batch_data['optim_lr'] == optim_lr].iloc[0][1]
    print(f"Yaml: {yaml_name}")
    print(f"Has been copied: {os.path.isdir(out_dir / yaml_name)}")

    if os.path.isdir(out_dir / yaml_name):
        continue

    checkpoint_iterations = [int(checkpoint.split('_')[-1].split('.')[0]) for checkpoint in checkpoints]

    maximum_checkpoint = max(checkpoint_iterations)
    print(f"{subfolder} has max checkpoint of {maximum_checkpoint}")

    if maximum_checkpoint == 4999:
        print("Proceeding to copy data")
        shutil.copytree(in_dir / subfolder, out_dir / yaml_name)
        print("Successfully copied")

        checkpoint_data = []
        for checkpoint in checkpoints:
            checkpoint_data.append(pd.read_csv(out_dir / yaml_name / "data" / checkpoint))

        combined_data = pd.concat(checkpoint_data, ignore_index=True)
        combined_data.drop(columns='Unnamed: 0', inplace=True)
        combined_data.to_csv(out_dir / yaml_name / "data" / f"{checkpoints[0][:len(checkpoints[0]) - 9]}.csv")

        for checkpoint in checkpoints:
            os.remove(out_dir / yaml_name / "data" / checkpoint)

    elif maximum_checkpoint > 1500:
        candidates_1500.append(batch_data.loc[batch_data['optim_lr'] == optim_lr].iloc[0][1])
        print("Added to 1500+ candidates")
    elif maximum_checkpoint > 3000:
        candidates_1500.append(batch_data.loc[batch_data['optim_lr'] == optim_lr].iloc[0][1])
        print("Added to 3000+ candidates")

print(f"1500 or more done: {candidates_1500}")
print(f"3000 or more done: {candidates_3000}")
