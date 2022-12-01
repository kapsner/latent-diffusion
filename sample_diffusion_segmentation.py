from genericpath import isdir
import numpy as np
from PIL import Image
import os
import pandas as pd
import shutil
from einops import rearrange
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.data.dce_mip import DCEMipMaskValidation


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("/raid/home/follels/Documents/latent-diffusion/configs/latent-diffusion/dce_mip-vq-seg.yaml")
    model = load_model_from_config(config, "/raid/home/follels/Documents/latent-diffusion/logs/2022-11-11T14-15-24_dce_mip-vq-seg/checkpoints/epoch=000163.ckpt")
    return model


model = get_model()
sampler = DDIMSampler(model)

number_of_samples_per_case = 10
ddim_steps = 200
ddim_eta = 1
batch_size = number_of_samples_per_case

validation_set = DCEMipMaskValidation()

save_path = "/raid/home/follels/Documents/latent-diffusion/samples/fake_segmentation"

with torch.no_grad():
    with model.ema_scope():
        for k in tqdm(range(len(validation_set))):
            val_sample = validation_set.__getitem__(k)
            samples = []
            c = model.get_learned_conditioning(
                val_sample[model.cond_stage_key].to(model.device)[None, None]
            )
            c = c.repeat(number_of_samples_per_case, 1, 1, 1)  # Repeat same segmentation mask conditioning
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            image = torch.clamp((val_sample["image"] + 1.0) / 2.0, min=0.0, max=1.0)
            segmentation = torch.clamp(val_sample["segmentation"] / 6.0, min=0.0, max=1.0)
            save_path_folder = os.path.join(save_path, f"{val_sample['info'].replace('.npy', '').split('/')[-1]}")
            if not os.path.exists(save_path_folder):
                os.mkdir(save_path_folder)
            img = Image.fromarray((255.0 * image).squeeze().cpu().numpy().astype(np.uint8))
            img.save(os.path.join(save_path_folder, "gt.png"))
            img = Image.fromarray((255.0 * segmentation).squeeze().cpu().numpy().astype(np.uint8))
            img.save(os.path.join(save_path_folder, "segmentation.png"))
            for i in range(x_samples_ddim.shape[0]):
                img = Image.fromarray((255.0 * x_samples_ddim[i]).squeeze().cpu().numpy().astype(np.uint8))
                img.save(os.path.join(save_path_folder, f"{i}.png"))
