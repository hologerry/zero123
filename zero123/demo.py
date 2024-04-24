import math
import os
import sys

import cv2
import numpy as np
import torch
import torchvision

from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from gradio_new import (
    CameraVisualizer,
    load_model_from_config,
    main_run,
    main_run_simple,
)
from ldm.util import create_carvekit_interface
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from tqdm import tqdm
from transformers import AutoFeatureExtractor  # , CLIPImageProcessor


def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = np.sqrt(xy + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])


def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond

    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T


def main_demo():
    device_idx = 3
    # ckpt = "/home/yuegao/Dynamics/stable-zero123/stable_zero123.ckpt"
    # ckpt = "/home/yuegao/Dynamics/zero123-weights/zero123-xl.ckpt"
    # ckpt = "/home/yuegao/Dynamics/zero123-weights/165000.ckpt"
    ckpt = "logs/2024-04-24T02-19-47_sd-scalar-flow-finetune-c_concat-256/checkpoints/last.ckpt"
    # config = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    config = "configs/sd-scalar-flow-finetune-c_concat-256.yaml"

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print("Instantiating LatentDiffusion...")
    models["turncam"] = load_model_from_config(config, ckpt, device=device)

    # cam_vis = CameraVisualizer(None)

    # img_path = "/home/yuegao/Dynamics/HyFluid/data/ScalarReal/colmap_100/input/train02.png"
    img_path = "/data/Dynamics/ScalarFlow/zero123_dataset/sim_000001_frame_000100/02.png"

    cond_cam_path = "/data/Dynamics/ScalarFlow/zero123_dataset/camera/02.npy"
    target_cam_path = "/data/Dynamics/ScalarFlow/zero123_dataset/camera/01.npy"
    cond_RT = np.load(cond_cam_path)
    target_RT = np.load(target_cam_path)

    d_T = get_T(target_RT, cond_RT)

    image = Image.open(img_path).convert("RGB")
    transforms = [
        torchvision.transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
    ]
    preprocess = torchvision.transforms.Compose(transforms)
    image = preprocess(image).unsqueeze(0).to(device)

    save_path = "outputs/vis_zero123_xl_fintune"

    os.makedirs(save_path, exist_ok=True)

    raw_im = cv2.imread(img_path)
    cv2.imwrite(f"{save_path}/raw_im.png", raw_im)

    # y_angle_values = [float(i) for i in range(-90, 91, 5)]

    # for idx, y in tqdm(enumerate(y_angle_values)):
    #     outputs = main_run(
    #         models,
    #         device,
    #         cam_vis,
    #         "trans_gen",
    #         0.0,
    #         y,
    #         0.0,
    #         raw_im=raw_im,
    #         preprocess=False,
    #         smoke=True,
    #         save_path=save_path,
    #     )
    #     # for o in outputs:
    #     #     print(type(o))

    #     out_imgs = outputs[-1]
    #     for i, img in enumerate(out_imgs):
    #         img.save(f"{save_path}/idx_{idx}_y{y}_output_{i}.png")

    out_imgs = main_run_simple(
        models,
        device,
        d_T,
        raw_im=raw_im,
        save_path=save_path,
    )
    # for o in outputs:
    #     print(type(o))

    for i, img in enumerate(out_imgs):
        img.save(f"{save_path}/idx_2to0_output_{i}.png")


if __name__ == "__main__":

    main_demo()
