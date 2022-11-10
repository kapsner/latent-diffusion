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


class DCEMip(Dataset):
    def __init__(self, path:str="/raid/store_your_files_here/dce_mip_diffusion/data/"):
        self.db = pd.read_csv(os.path.join(path, "dataset.csv"))
        # self.db["birads_max"] = self.db[["birads_li", "birads_re"]].max(1)
        # self.db = self.db[self.db["birads_max"] > -1]
        self.db = self.db[self.db["birads_li"] > -1]
        self.db = self.db[self.db["birads_re"] > -1]
        # self.db["birads_binary"] = self.db["birads_max"] > 3
        path_images = os.path.join(path, "dce_mips2")
        files = [os.path.join(path_images, file) for file in os.listdir(path_images)]
        self.files = [file for file in files if ".npy" in file]

    def __getitem__(self, index):
        row = self.db.iloc[index]
        image = np.load(self.files[index]).squeeze()
        image = TF.to_tensor(image).squeeze()

        birads_li = torch.tensor(int(row["birads_li"]))
        birads_re = torch.tensor(int(row["birads_re"]))
        
        class_label = torch.cat((torch.nn.functional.one_hot(birads_li, 7),
                                 torch.nn.functional.one_hot(birads_re, 7))).squeeze()
        human_label = f"BIRADS links {birads_li}, rechts {birads_re}"

        sample = {"image": image, "class_label": class_label, "human_label": human_label}

        return sample


class DCEMipTrain(DCEMip):
    def __init__(self):
        super().__init__()
        self.files = self.files[:int(0.7 * len(self.files))]
        print(f"Size of training dataset: {len(self.files)}.")

    def __len__(self):
        return len(self.files)


class DCEMipValidation(DCEMip):
    def __init__(self):
        super().__init__()
        self.files = self.files[int(0.7 * len(self.files)):int(0.9 * len(self.files))]
        print(f"Size of validation dataset: {len(self.files)}.")

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as T
    ds = DCEMipTrain()
    transform = T.ToPILImage()
    save_path = "/raid/home/follels/Documents/latent-diffusion/samples/real"
    for i in range(100):
        sample = ds.__getitem__(i)
        image = sample["image"]
        condition = sample["class_label"]
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        # plt.imshow(image.squeeze())
        # plt.savefig(f"{save_path}/sample_{i}.png")
        im = transform(image)
        im.save(f"{save_path}/sample_{i}.png")
        

    # Test embedding
    # from ldm.modules.encoders.modules import ClassEmbedder
    # emb = ClassEmbedder(512, 14)
    # print(emb(sample, key="class_label").shape)
    # print(emb(sample, key="class_label"))

