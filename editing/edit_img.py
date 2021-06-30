from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path[0] = "."
from train_pre import preprocess_images

parser = ArgumentParser()
parser.add_argument("--infile", type=str, required=True)
parser.add_argument("--outfile", type=str, required=True)
parser.add_argument("-d", "--distortion", action="append")
args = parser.parse_args()

#print("possible image editing steps:", preprocess_images.mapping["all_changes"])

print(f"reading from\t{args.infile}")
print(f"saving to\t{args.outfile}")

selected_distortions = [(d.split(",")[0], int(d.split(",")[1])) for d in args.distortion]
print(f"selected distortions:\t{selected_distortions}")

ed = preprocess_images.ImageEditor()

res = ed.distort_list_image(path=args.infile, distortion_intens_tuple_list=selected_distortions)

for k, v in res.items():
    save_path = Path(args.outfile).parent / f"{Path(args.outfile).stem}_{k}{Path(args.infile).suffix}"
    print(f"saving to\t{save_path}")
    v.save(save_path)
