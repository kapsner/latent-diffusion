import os, shutil
import pandas as pd
import numpy as np
import math
import torch
import zipfile
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ChexpertBase(Dataset):
    def __init__(self, path:str="/lustre/iwi5/iwi5044h/CheXpert-v1.0-small.zip"):
        self.local_path = "/scratch/CheXpert-v1.0-small"
        if not os.path.isdir(self.local_path):
            os.mkdir(self.local_path)
            print(f"INFO: Copying from {path} to {self.local_path + '.zip'}. This might take some time.")
            shutil.copy(path, self.local_path + ".zip")
            print(f"INFO: Unzipping {self.local_path + '.zip'}. This might take some time.")
            with zipfile.ZipFile(self.local_path + '.zip') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member, self.local_path)
                    except zipfile.error as e:
                        pass
        self.local_path = "/scratch/CheXpert-v1.0-small/CheXpert-v1.0-small"
        self.db = None

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = os.path.join(os.path.dirname(self.local_path), row["Path"])
        with Image.open(path) as im:
            image = TF.to_tensor(im)
        image = TF.resize(image, (320, 384))

        image = 2 * (image - image.min()) / (image.max() - image.min()) - 1

        image = image.squeeze()

        condition_age = int(row["Age"])
        condition_view = defaultdict(float, {"Frontal": 1, "Lateral": 2})[row["Frontal/Lateral"]]
        condition_appa = defaultdict(float, {"AP": 1, "PA": 2})[row["AP/PA"]]
        condition_pneumonia = 1 + int(row["Pneumonia"]) if not math.isnan(row["Pneumonia"]) else 0
        condition_lung_opacity = 1 + int(row["Lung Opacity"]) if not math.isnan(row["Lung Opacity"]) else 0
        condition = torch.tensor([[condition_age, condition_view, condition_appa, condition_pneumonia, condition_lung_opacity]]).long()
        
        sample = {"image": image, "condition": condition}

        return sample


class ChexpertTrain(ChexpertBase):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv(os.path.join(self.local_path, "train.csv"))
    
    def __len__(self):
        return len(self.df)


class ChexpertValidation(ChexpertBase):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv(os.path.join(self.local_path, "valid.csv"))
    
    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    import torch.nn as nn
    ds = ChexpertTrain()
    sample = ds.__getitem__(0)
    image = sample["image"]
    condition = sample["condition"]
    print(image.shape)
    print(image.mean())
    print(image.std())
    print(condition.shape)
    print(condition)

    # Test embedding
    from ldm.modules.encoders.modules import MultiClassEmbedder
    emb = MultiClassEmbedder(512)
    print(emb({"condition": condition}, key="condition").shape)
