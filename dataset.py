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


class AVA:
    def __init__(
        self,
        mode: str,
        image_dir: str = str(Path.home()) + '\\Documents\\datasets\\images\\AVA\\images',
        percentage_of_dataset: int = None,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.mode = mode
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset

        self.files = pd.read_csv(f"analysis\\sets\\{mode}_set.csv")
        if self.percentage_of_dataset is not None:
            self.files = self.files[: int(len(self.files) * (self.percentage_of_dataset / 100))]

        logging.info(f"found {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = self.image_dir + '\\' + str(int(self.files.iloc[idx][0])) + ".jpg"
        pil_img = Image.open(path).convert("RGB")
        #img = transforms.ToTensor()(img)
        #if self.normalize:
        #    pil_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(pil_img)
        rating = self.files.iloc[idx][1]
        return {"image_id": int(self.files.iloc[idx][0]), "ava_score": rating, "img": pil_img}


class Pexels:
    def __init__(
        self,
        mode: str,
        image_dir: str = 'datasets\\pexels\\images',
        percentage_of_dataset: int = None,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.mode = mode
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset

        self.files = pd.read_csv(f"analysis\\sets\\{mode}_set.csv")
        if self.percentage_of_dataset is not None:
            self.files = self.files[: int(len(self.files) * (self.percentage_of_dataset / 100))]

        logging.info(f"found {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            print("except"  + str(idx))
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = self.image_dir + '\\' + str(self.files.iloc[idx][0])
        pil_img: Image = Image.open(path).convert("RGB")
        return {"image_id": self.files.iloc[idx][0], "img": pil_img}


class Landscapes:
    def __init__(
        self,
        mode: str,
        image_dir: str = str(Path.home()) + '\\Documents\\datasets\\images\\Landscapes\\images',
        percentage_of_dataset: int = None,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.mode = mode
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset

        self.files = pd.read_csv(f"analysis\\sets\\{mode}_set.csv")
        if self.percentage_of_dataset is not None:
            self.files = self.files[: int(len(self.files) * (self.percentage_of_dataset / 100))]

        logging.info(f"found {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = self.image_dir + '\\' + str(int(self.files.iloc[idx][0])) + ".jpg"
        pil_img = Image.open(path).convert("RGB")
        return {"image_id": int(self.files.iloc[idx][0]), "img": pil_img}

class LandscapesTop:
    def __init__(
        self,
        mode: str,
        image_dir: str = str(Path.home()) + '\\Documents\\datasets\\images\\Landscapes\\images',
        percentage_of_dataset: int = None,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.mode = mode
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset

        self.files = pd.read_csv(f"analysis\\sets\\{mode}_set.csv")
        if self.percentage_of_dataset is not None:
            self.files = self.files[: int(len(self.files) * (self.percentage_of_dataset / 100))]

        logging.info(f"found {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = self.image_dir + '\\' + str(int(self.files.iloc[idx][0])) + ".jpg"
        pil_img = Image.open(path).convert("RGB")
        return {
            "image_id": int(self.files.iloc[idx][0]), "orig_ia_pre_score": float(self.files.iloc[idx][1]),
            "orig_ia_pre_styles_change": self.files.iloc[idx][2],
            "img": pil_img
                }
