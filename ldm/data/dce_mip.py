import os, shutil
import pandas as pd
import numpy as np
import math
from glob import glob
import json
import torch
import zipfile
from tqdm import tqdm
from scipy.ndimage import zoom
import SimpleITK as sitk
from collections import defaultdict
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold


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


class DCEMipMask(Dataset):
    def __init__(self, path:str="/raid/store_your_files_here/dce_mip_diffusion/data/"):
        label_path = os.path.join(path, "labelsTr")
        label_jsons = glob(os.path.join(label_path, "*.json"))
        print(f"# label: {len(label_jsons)}")
        label_nii = [l.replace("json", "nii.gz") for l in label_jsons]
        print(f"# nifti: {len(label_nii)}")
        mip_files = [l.replace("labelsTr", "dce_mips2").replace(".json", "_mip.npy") for l in label_jsons]
        print(f"# mips: {len(mip_files)}")
        label_jsons, label_nii, mip_files, birads_max = self.remove_missing_files((label_jsons, label_nii, mip_files))

        database = pd.DataFrame({
            "label_json_path": label_jsons,
            "label_nii_path": label_nii,
            "mip_path": mip_files,
            "birads_max": birads_max
        })
        print(f"# database: {len(database)}")

        pattern_extract = database.mip_path.str.extract(
            pat=".*/(\d+)_(\d+)_(\d+).*\.npy|nii\.gz$"
        )

        pattern_extract.rename(
            columns={
                0: "patient_id",
                1: "study_date",
                2: "series_num"
            },
            inplace=True
        )
        database = database.merge(
            right=pattern_extract,
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=["", ""]
        )



        splits = {}

        kfold = GroupKFold(n_splits=4)

        # split into training / test dataset here
        for _i, (_train_idx, _test_idx) in enumerate(kfold.split(database[["birads_max"]], groups=database[["patient_id"]])):
            splits[_i] = {
                "train_idx": _train_idx,
                "test_idx": _test_idx
            }
        self.db_train = database.iloc[splits[1]["train_idx"]]
        self.db_val =  database.iloc[splits[1]["test_idx"]]
        
        

    @staticmethod
    def remove_missing_files(files:tuple):
        new_files = []
        for lbl_json, lbl_nifti, mip_file in zip(*files):
            if os.path.exists(lbl_json) and \
              os.path.exists(lbl_nifti) and \
              os.path.exists(mip_file):
                with open(lbl_json) as f:
                    label_json = json.load(f)
                    if len(label_json["instances"]) > 0:
                        birads_max = pd.DataFrame.from_dict(label_json, orient="columns").instances.max()
                    else:
                        birads_max = 1
                new_files.append((lbl_json, lbl_nifti, mip_file, birads_max))
            else:
                print(f"json: {lbl_json} \nnifti: {lbl_nifti} \nmip: {mip_file}")
        new_files = zip(*new_files)
        return new_files

    def __getitem__(self, index):
        row = self.db.iloc[index]

        image = np.load(row["mip_path"]).squeeze()
        image = TF.to_tensor(image).squeeze()

        with open(row["label_json_path"]) as f:
            label_json = json.load(f)

        mask = np.zeros_like(image)
        if len(label_json["instances"]) > 0:
            mask = sitk.ReadImage(row["label_nii_path"])
            mask = sitk.GetArrayFromImage(mask)
            mask = mask[0] 

            mapper = dict(zip([int(k) for k in label_json["instances"].keys()], label_json["instances"].values()))
            mapper[0] = 0  # Background
            mask = np.vectorize(mapper.__getitem__ )(mask).astype(np.float32)
            mask = zoom(mask, (image.shape[0] / mask.shape[0], image.shape[1] / mask.shape[1]), order=0)
        mask = TF.to_tensor(mask).squeeze()

        sample = {"image": image, "segmentation": mask, "info": row["mip_path"]}
        return sample


class DCEMipMaskTrain(DCEMipMask):
    def __init__(self, tocsv: bool = False):
        super().__init__()
        #self.db = self.db.loc[:int(0.7 * len(self.db)), :].copy()
        self.db = self.db_train.copy()
        print(f"Size of training dataset: {len(self.db)}.")
        
        if tocsv:
            self.db.to_csv(
                path_or_buf="/home/user/development/trainings/diffusionmodels/train.csv",
                index=False
            )

    def __len__(self):
        return len(self.db)


class DCEMipMaskValidation(DCEMipMask):
    def __init__(self, tocsv: bool = False):
        super().__init__()
        #self.db = self.db.loc[int(0.7 * len(self.db)):int(0.9 * len(self.db)), :].copy()
        self.db = self.db_val.copy()
        print(f"Size of validation dataset: {len(self.db)}.")
        
        if tocsv:
            self.db.to_csv(
                path_or_buf="/home/user/development/trainings/diffusionmodels/val.csv",
                index=False
            )

    def __len__(self):
        return len(self.db)


def dataset_stats(ds):
    unique = {}
    for i in range(len(ds)):
        item = ds.__getitem__(i)
        for u in list(item["segmentation"].unique()):
            if int(u) not in unique.keys():
                unique[int(u)] = 1    
            else:
                unique[int(u)] += 1
    print(unique)

def target_class_stats(ds):
    print(ds.db.birads_max.value_counts(normalize=True) * 100)

if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as T
    ds_train = DCEMipMaskTrain(tocsv=True)
    target_class_stats(ds_train)
    
    ds_val = DCEMipMaskValidation(tocsv=True)
    target_class_stats(ds_val)
    # transform = T.ToPILImage()
    # save_path = "/raid/home/follels/Documents/latent-diffusion/samples/real"
    # for i in range(100):
    #     sample = ds.__getitem__(i)
    #     image = sample["image"]
    #     condition = sample["class_label"]
    #     image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    #     # plt.imshow(image.squeeze())
    #     # plt.savefig(f"{save_path}/sample_{i}.png")
    #     im = transform(image)
    #     im.save(f"{save_path}/sample_{i}.png")
        

    # Test embedding
    # from ldm.modules.encoders.modules import ClassEmbedder
    # emb = ClassEmbedder(512, 14)
    # print(emb(sample, key="class_label").shape)
    # print(emb(sample, key="class_label"))
