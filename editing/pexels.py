import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

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
        pil_img: Image = Image.open(path).convert("RGB")
        return {"image_id": self.files.iloc[idx][0], "img": pil_img}