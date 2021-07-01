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
        path = Path('dataset_processing/pexels_test_set_3500.csv')
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
print("Loaded Pexels dataset")
ed = preprocess_images.ImageEditor()

for i in range(len(pexels)):  # len(pexels)
    print("editing: " + pexels.__getitem__(i) + 'with')
    distortions1 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                    ('shadows', randrange(-1, 4)), ('highlights', randrange(-4, 1)), ('exposure', randrange(-2, 1))]
    while distortions1[0][1] == distortions1[1][1] == distortions1[2][1] == distortions1[3][1] == distortions1[4][1] == 0:
        distortions1 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                        ('shadows', randrange(-1, 3)), ('highlights', randrange(-3, 1)), ('exposure', randrange(-1, 1))]

    print(distortions1)

    image = Image.open('/' + pexels.__getitem__(i)).convert("RGB")

    for distortion in distortions1:
        image = ed.distort_image(distortion[0], distortion[1], img=image)

    save_path = Path('/out/') / \
                    f"{pexels.__getitem__(i).split('/')[1].split('.')[0]}_{distortions1[0][1]}_{distortions1[1][1]}_" \
                    f"{distortions1[2][1]}_{distortions1[3][1]}_{distortions1[4][1]}." \
                    f"{pexels.__getitem__(i).split('/')[1].split('.')[1]}"
    print(f"saving to\t{save_path}")
    image.save(save_path)

    print("editing: " + pexels.__getitem__(i) + 'with')
    distortions2 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                    ('shadows', randrange(-1, 4)), ('highlights', randrange(-4, 1)), ('exposure', randrange(-2, 1))]
    while distortions2[0][1] == distortions2[1][1] == distortions2[2][1] == distortions2[3][1] == distortions2[4][
        1] == 0 or distortions1 == distortions2:
        distortions2 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                        ('shadows', randrange(-1, 3)), ('highlights', randrange(-3, 1)), ('exposure', randrange(-1, 1))]

    print(distortions2)

    image = Image.open('/' + pexels.__getitem__(i)).convert("RGB")

    for distortion in distortions2:
        image = ed.distort_image(distortion[0], distortion[1], img=image)

    save_path = Path('/out/') / \
                f"{pexels.__getitem__(i).split('/')[1].split('.')[0]}_{distortions2[0][1]}_{distortions2[1][1]}_" \
                f"{distortions2[2][1]}_{distortions2[3][1]}_{distortions2[4][1]}." \
                f"{pexels.__getitem__(i).split('/')[1].split('.')[1]}"
    print(f"saving to\t{save_path}")
    image.save(save_path)

    print("editing: " + pexels.__getitem__(i) + 'with')
    distortions3 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                    ('shadows', randrange(-1, 4)), ('highlights', randrange(-4, 1)), ('exposure', randrange(-2, 1))]
    while distortions3[0][1] == distortions3[1][1] == distortions3[2][1] == distortions3[3][1] == distortions3[4][
        1] == 0 or distortions1 == distortions3 or distortions2 == distortions3:
        distortions3 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                        ('shadows', randrange(-1, 3)), ('highlights', randrange(-3, 1)), ('exposure', randrange(-1, 1))]

    print(distortions3)

    image = Image.open('/' + pexels.__getitem__(i)).convert("RGB")

    for distortion in distortions3:
        image = ed.distort_image(distortion[0], distortion[1], img=image)

    save_path = Path('/out/') / \
                f"{pexels.__getitem__(i).split('/')[1].split('.')[0]}_{distortions3[0][1]}_{distortions3[1][1]}_" \
                f"{distortions3[2][1]}_{distortions3[3][1]}_{distortions3[4][1]}." \
                f"{pexels.__getitem__(i).split('/')[1].split('.')[1]}"
    print(f"saving to\t{save_path}")
    image.save(save_path)

    print("editing: " + pexels.__getitem__(i) + 'with')
    distortions4 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                    ('shadows', randrange(-1, 4)), ('highlights', randrange(-4, 1)), ('exposure', randrange(-2, 1))]
    while distortions4[0][1] == distortions4[1][1] == distortions4[2][1] == distortions4[3][1] == distortions4[4][
        1] == 0 or distortions1 == distortions4 or distortions2 == distortions4 or distortions3 == distortions4:
        distortions4 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                        ('shadows', randrange(-1, 3)), ('highlights', randrange(-3, 1)), ('exposure', randrange(-1, 1))]

    print(distortions4)

    image = Image.open('/' + pexels.__getitem__(i)).convert("RGB")

    for distortion in distortions4:
        image = ed.distort_image(distortion[0], distortion[1], img=image)

    save_path = Path('/out/') / \
                f"{pexels.__getitem__(i).split('/')[1].split('.')[0]}_{distortions4[0][1]}_{distortions4[1][1]}_" \
                f"{distortions4[2][1]}_{distortions4[3][1]}_{distortions4[4][1]}." \
                f"{pexels.__getitem__(i).split('/')[1].split('.')[1]}"
    print(f"saving to\t{save_path}")
    image.save(save_path)