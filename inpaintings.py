import argparse, os, sys, glob
from omegaconf import OmegaConf
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import torch
#from main import instantiate_from_config
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.dce_mip import DCEMipMask as dm
import json
from scipy.ndimage import zoom
from PIL import Image
import torchvision.transforms.functional as TF
from ldm.modules.encoders.modules import SpatialRescaler

gpu_device = "cuda:1"
ddim_steps = 200
ddim_eta = 1

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu_device)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(gpu_device)
    model.eval()
    return model


def get_model(config_file, checkpoint_path):
    config = OmegaConf.load(config_file)
    model = load_model_from_config(config, checkpoint_path)
    return model


def make_batch(image_path, mask_path, device):
    image = np.load(image_path).squeeze()
    img_shape = image.shape

    mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask)
    mask = mask[0]
    mask = mask.astype(np.float32)

    # transpose mask to image shape
    mask = zoom(mask, (img_shape[0] / mask.shape[0], img_shape[1] / mask.shape[1]), order=0)

    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0
    
    masked_image = np.copy(image)
    masked_image[mask > 0] = -1.0

    image = torch.from_numpy(image).squeeze()
    mask = torch.from_numpy(mask).squeeze()
    masked_image = torch.from_numpy(masked_image).squeeze()

    # img = Image.fromarray((255.0 * masked_image).squeeze().cpu().numpy().astype(np.uint8))
    # img = Image.fromarray((255.0 * (1-mask)).squeeze().astype(np.uint8))
    # img = Image.fromarray((255.0 * image).squeeze().astype(np.uint8))
    # img.save(outpath.replace(".npy", "_mask.png"))

    batch = {"image": image, "segmentation": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        #batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    label_path = os.path.join(opt.indir, "labelsTr")
    label_jsons = glob.glob(os.path.join(label_path, "*.json"))
    label_nii = [l.replace("json", "nii.gz") for l in label_jsons]
    mip_files = [l.replace("labelsTr", "dce_mips2").replace(".json", "_mip.npy") for l in label_jsons]
    label_jsons, label_nii, mip_files, birads_max = dm().remove_missing_files((label_jsons, label_nii, mip_files))

    masks = []
    images = []

    for _mask, _image, _label in zip(label_nii, mip_files, label_jsons):
        with open(_label) as f:
            label_json = json.load(f)
        if len(label_json["instances"]) > 0:
            masks += [_mask]
            images += [_image]
    print(f"# images: {len(images)}")
    print(f"# masks: {len(masks)}")

    model = get_model(
        config_file="models/ldm/inpainting_big/config_dce_mip.yaml",
        # autoencoder --> wrong
        # checkpoint_path="/raid/store_your_files_here/dce_mip_diffusion/models/ae_dce_mip/checkpoints/epoch=000188.ckpt"
        # ldm
        checkpoint_path=os.path.join(
            "/home/user/development/trainings/diffusionmodels",
            "2023-01-21T04-01-09_dce_mip-vq-seg_1204/checkpoints/val",
            "loss=0.122759-epoch=000375.ckpt"
        )
    )
    our_weights = True

    # model = get_model(
    #     config_file="models/ldm/inpainting_big/config.yaml",
    #     checkpoint_path="models/ldm/inpainting_big/last.ckpt"
    # )
    # our_weights = False
    
    #config = OmegaConf.load("models/ldm/inpainting_big/config_dce_mip.yaml")
    #model = instantiate_from_config(config.model)
    # model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
    #                       strict=False)
    sampler = DDIMSampler(model)

    rescaler = SpatialRescaler(in_channels=4, out_channels=3, multiplier=1)
    rescaler.to(gpu_device)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=gpu_device)

                if our_weights:
                    #input_img = batch["masked_image"][None, None]
                    input_img = batch["masked_image"][None, None]
                else:
                    input_img = batch["masked_image"][None].repeat(3, 1, 1)[None]

                # encode masked image
                #c = model.cond_stage_model.encode(input_img)
                # https://github.com/CompVis/latent-diffusion/blob/2b46bcb98c8e8fdb250cb8ff2e20874f3ccdd768/ldm/models/diffusion/ddpm.py#L660
                c = model.encode_first_stage(input_img)
                z = model.get_first_stage_encoding(c).detach()
                cc = torch.nn.functional.interpolate(batch["segmentation"][None, None],
                                                     size=c.shape[-2:])
                # c = torch.cat((c, cc), dim=1)
                # c = rescaler.encode(c)

                #shape = (c.shape[1]-1,)+c.shape[2:]
                N = c.shape[0]
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=N,
                    shape=[3, 64, 64],
                    verbose=False,
                    mask=cc,
                    x0=z[:N],
                    eta=ddim_eta
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                #x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
                # mask = torch.clamp((batch["mask"]+1.0)/2.0, min=0.0, max=1.0)
                mask = torch.clamp(batch["segmentation"], min=0.0, max=1.0)
                masked = torch.clamp((batch["masked_image"]+1.0)/2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                if not our_weights:
                    predicted_image = predicted_image[0, 0, : ]

                #inpainted = (1-mask) * image + mask * predicted_image
                inpainted = masked + (mask * predicted_image)

                
                img = Image.fromarray((255.0 * inpainted).squeeze().cpu().numpy().astype(np.uint8))
                img.save(outpath.replace(".npy", "_inpaint.png"))

                img = Image.fromarray((255.0 * predicted_image).squeeze().cpu().numpy().astype(np.uint8))
                img.save(outpath.replace(".npy", "_pred.png"))

                img = Image.fromarray((255.0 * mask).squeeze().cpu().numpy().astype(np.uint8))
                img.save(outpath.replace(".npy", "_mask.png"))

                img = Image.fromarray((255.0 * masked).squeeze().cpu().numpy().astype(np.uint8))
                img.save(outpath.replace(".npy", "_masked_img.png"))

                img = Image.fromarray((255.0 * image).squeeze().cpu().numpy().astype(np.uint8))
                img.save(outpath.replace(".npy", "_original.png"))
