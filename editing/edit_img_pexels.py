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
    ):
        self.image_dir = image_dir
        path = Path('dataset_processing/pexels_test_set_2039.csv')
        df = pd.read_csv(path)
        self.files = df.sample(n=5000)

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

for i in range(len(pexels)):
    print("editing: " + pexels.__getitem__(i) + 'with')
    distortions1 = [('saturation', randrange(-1, 1)), ('contrast', randrange(-1, 1)),
                    ('shadows', randrange(-1, 4)), ('highlights', randrange(-4, 1)), ('exposure', randrange(-2, 1))]
    while distortions1[0][1] == distortions1[1][1] == distortions1[2][1] == distortions1[3][1] == distortions1[4][
        1] == 0:
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
