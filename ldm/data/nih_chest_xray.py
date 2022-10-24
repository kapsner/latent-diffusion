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


class NIHBase(Dataset):
    def __init__(self, path:str="/lustre/iwi5/iwi5044h/data.zip"):
        self.local_path = "/scratch/nih"
        if not os.path.isdir(self.local_path):
            os.mkdir(self.local_path)
            shutil.copy(os.path.join(os.path.dirname(path), "reduced_nih_labels.csv"), os.path.join(self.local_path, "reduced_nih_labels.csv"))
            print(f"INFO: Copying from {path} to {self.local_path + '.zip'}. This might take some time.")
            shutil.copy(path, self.local_path + ".zip")
            print(f"INFO: Unzipping {self.local_path + '.zip'}. This might take some time.")
            with zipfile.ZipFile(self.local_path + '.zip') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member, self.local_path)
                    except zipfile.error as e:
                        pass
        self.db = pd.read_csv(os.path.join(self.local_path, "reduced_nih_labels.csv"))
        labels = list(self.db["Finding_Labels"].unique())
        labels.sort()
        unique_labels_mapper = dict(zip(labels, range(len(labels))))
        self.unique_labels_mapper_inverse = dict(zip(range(len(labels)), labels))
        self.db["Finding_Labels_mapped"] = self.db["Finding_Labels"].map(unique_labels_mapper)
        image_folders = [f for f in os.listdir(self.local_path) if "images_" in f]
        image_paths = []
        for folder in image_folders:
            images = os.listdir(os.path.join(self.local_path, folder, "images"))
            images = [os.path.join(self.local_path, folder, "images", image) for image in images]
            image_paths.extend(images)
        path_db = pd.DataFrame(image_paths, columns=["Path"])
        path_db["Image_Index"] = path_db["Path"].str.split("/").str[-1]
        self.db = self.db.merge(path_db, on="Image_Index")


    def __getitem__(self, index):
        row = self.db.iloc[index]
        path = row["Path"]
        with Image.open(path) as im:
            image = TF.to_tensor(im)
        image = TF.resize(image, (256, 256))

        image = 2 * (image - image.min()) / (image.max() - image.min()) - 1

        image = image.squeeze()
        if len(image.shape) > 2:
            image = image[0]
        
        # TODO: just temporarily for work with pretrained AE
        # image = torch.stack([image, image, image], -1)

        class_label = torch.tensor(int(row["Finding_Labels_mapped"]))
        human_label = self.unique_labels_mapper_inverse[int(class_label)]

        sample = {"image": image, "class_label": class_label, "human_label": human_label}

        return sample


class NIHTrain(NIHBase):
    def __init__(self):
        super().__init__()
        self.df = self.db[self.db["fold"] == "train"]
        print(f"Number of training samples ({len(self.df)})")

    def __len__(self):
        return len(self.df)


class NIHValidation(NIHBase):
    def __init__(self):
        super().__init__()
        self.df = self.db[self.db["fold"] == "val"]
        print(f"Number of validation samples ({len(self.df)})")

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    import torch.nn as nn
    ds = NIHTrain()
    sample = ds.__getitem__(0)
    image = sample["image"]
    condition = sample["class_label"]
    print(image.shape)
    print(image.mean())
    print(image.std())
    print(condition)

    # Test embedding
    from ldm.modules.encoders.modules import ClassEmbedder
    emb = ClassEmbedder(512, 15)
    print(emb(sample, key="class_label").shape)
    print(emb(sample, key="class_label"))
