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
from taming.models import vqgan


def unique_label_mapper_and_inverse():
    db = pd.read_csv(os.path.join("/lustre/iwi5/iwi5044h", "reduced_nih_labels.csv"))
    labels = list(db["Finding_Labels"].unique())
    return dict(zip(range(len(labels)), labels)), dict(zip(labels, range(len(labels))))


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
    config = OmegaConf.load("/home/hpc/iwi5/iwi5044h/latent-diffusion/configs/latent-diffusion/nih-vq.yaml")  
    model = load_model_from_config(config, "/home/hpc/iwi5/iwi5044h/latent-diffusion/logs/2022-10-09T16-37-06_nih-vq/checkpoints/last.ckpt")
    return model


model = get_model()
sampler = DDIMSampler(model)

classes = np.arange(15)   # define classes to be sampled here
label_mapper_inverse, label_mapper = unique_label_mapper_and_inverse()
# n_samples_per_class = 100

ddim_steps = 200
ddim_eta = 1
scale = 3.0   # for unconditional guidance
batch_size = 128

# all_samples = list()
# save_path = "/home/hpc/iwi5/iwi5044h/latent-diffusion/samples_eta_1_200/"
save_path = "/home/woody/iwi5/shared/Diffusion/images"
save_path_local = "/scratch/images"
db = pd.read_csv("/lustre/iwi5/iwi5044h/reduced_nih_labels.csv")
db["labels_mapped"] = db["Finding_Labels"].map(label_mapper)

with torch.no_grad():
    with model.ema_scope():        
        for class_label in tqdm(classes):
            samples_per_class = len(db[db["labels_mapped"] == class_label])
            names = db[db["labels_mapped"] == class_label]["Image_Index"].tolist()
            
            number_of_steps = samples_per_class // batch_size
            remainder = samples_per_class % batch_size
            batch_sizes = [batch_size] * number_of_steps
            batch_sizes.append(remainder)
            samples = []
            for n_samples_per_class in tqdm(batch_sizes, leave=False):
                # print(f"rendering {n_samples_per_class} examples of class '{label_mapper_inverse[class_label]}' in {ddim_steps} steps and using s={scale:.2f}.")
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class*[15]).to(model.device)}
                )
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                samples.append(x_samples_ddim)

            if not os.path.exists(save_path_local):
                os.mkdir(save_path_local)
            samples = torch.cat(samples, 0)
            for name, sample in zip(names, samples):
                img = Image.fromarray((255.0 * sample).squeeze().cpu().numpy().astype(np.uint8))
                img.save(os.path.join(save_path_local, name))

print("Creating archive from images.")
shutil.make_archive("/scratch/images", 'zip', save_path_local)
print("Copying to woody.")
shutil.copy("/scratch/images.zip", "/home/woody/iwi5/shared/Diffusion/images.zip")


# # display as grid
# grid = torch.stack(all_samples, 0)
# grid = rearrange(grid, 'n b c h w -> (n b) c h w')
# grid = make_grid(grid, nrow=n_samples_per_class)

# # to image
# grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
# img = Image.fromarray(grid.astype(np.uint8))
# img.save("/home/hpc/iwi5/iwi5044h/latent-diffusion/samples/samples.png")

