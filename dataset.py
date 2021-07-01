import logging
import random
import os

import pandas as pd
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
        image_dir: Path = Path('datasets/pexels/images'),
        percentage_of_dataset: int = None,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.mode = mode
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset
        path = Path('analysis/sets/')
        path = path / (mode + '_set.csv')
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

class Pexels_hyperparamsearch:
    def __init__(
        self,
        image_orig_dir: Path = Path('datasets/pexels_dist/images'),
        image_dist_dir: Path = Path('datasets/pexels_dist/images'),
        sample_size: int = 5000,
        horizontal_flip: bool = False,
        normalize: bool = False,
    ):
        self.image_orig_dir = image_orig_dir
        self.image_dist_dit = image_dist_dir
        self.normalize = normalize
        self.horizontal_flip = horizontal_flip
        photos = os.listdir(image_dist_dir)
        df = pd.DataFrame(data={'images': photos})
        self.files = df.sample(n=sample_size)

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
        photo = str(self.files.iloc[idx][0])
        path_dist = self.image_dist_dit / photo
        print(f"path_dist: {path_dist}")
        path_orig = self.image_orig_dir / f"{photo.split('_')[0]}.{photo.split('.')[-1]}"
        print(f"path_orig: {path_orig}")
        img_orig: Image = Image.open(path_orig).convert("RGB")
        img_dist: Image = Image.open(path_dist).convert("RGB")

        return {"image_id": self.files.iloc[idx][0], "img_orig": img_orig, "img_dist": img_dist}
