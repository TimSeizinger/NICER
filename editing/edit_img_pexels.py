import logging
import pandas as pd
from PIL import Image
from pathlib import Path
from random import randrange
import sys
sys.path[0] = "."
from train_pre import preprocess_images


class Pexels:
    def __init__(
        self,
        image_dir: Path = Path('dataset/'),
        percentage_of_dataset: int = None,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset
        path = Path('scratch/dataset_processing/pexels_test_set.csv')
        self.files = pd.read_csv(path)
        if self.percentage_of_dataset is not None:
            self.files = self.files[: int(len(self.files) * (self.percentage_of_dataset / 100))]

        logging.info(f"found {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        '''
        try:
            return self._actualgetitem(idx)
        except:
            print("except" + str(idx))
            return self[random.randint(0, len(self))]
        '''
        return self._actualgetitem(idx)

    def _actualgetitem(self, idx: int):
        path = self.image_dir / str(self.files.iloc[idx][0])
        return str(path)


pexels = Pexels()
ed = preprocess_images.ImageEditor()

for i in range(10):  # len(pexels)
    distortions = [('contrast', randrange(-1, 1)), ('saturation', randrange(-1, 1)), ('exposure', randrange(-1, 1)),
                   ('shadows', randrange(-1, 3)), ('highlights', randrange(-3, 1))]

    result = ed.distort_list_image(distortions, pexels.__getitem__(i))

    for k, v in result.items():
        save_path = Path('out/') / f"{Path(pexels.__getitem__(i)).prefix}_edited_{Path(pexels.__getitem__(i)).suffix}"
        print(f"saving to\t{save_path}")
        v.save(save_path)