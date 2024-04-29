import math
import os

import cv2
import numpy as np
import torch
import torchvision

from camera_utils import get_T
from demo_helpers import main_run_simple
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from utils import load_model_from_config


def main_demo(device_idx=0, finetune_step=6000):
    # ckpt = "logs/2024-04-24T02-19-47_sd-scalar-flow-finetune-c_concat-256/checkpoints/last.ckpt"
    ckpt = f"logs/2024-04-24T14-04-16_sd-scalar-flow-finetune-c_concat-256/checkpoints/step={finetune_step-1:09d}.ckpt"
    # ckpt = f"logs/2024-04-24T14-04-16_sd-scalar-flow-finetune-c_concat-256/checkpoints/last.ckpt"
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

    scalar_real_zero123_dataset_path = "/data/Dynamics/ScalarReal/zero123_dataset"

    frame_id = 119
    source_cam = "02"
    # target_cam = "00"
    for target_cam in ["00", "01", "03", "04"]:
        cond_img_path = f"{scalar_real_zero123_dataset_path}/frame_{frame_id:06d}/{source_cam}.png"
        gt_img_path = f"{scalar_real_zero123_dataset_path}/frame_{frame_id:06d}/{target_cam}.png"
        assert os.path.exists(cond_img_path), f"cond_img_path {cond_img_path} does not exist"
        assert os.path.exists(gt_img_path), f"gt_img_path {gt_img_path} does not exist"

        cond_cam_path = f"{scalar_real_zero123_dataset_path}/camera/{source_cam}.npy"
        target_cam_path = f"{scalar_real_zero123_dataset_path}/camera/{target_cam}.npy"
        cond_RT = np.load(cond_cam_path)
        target_RT = np.load(target_cam_path)

        d_T = get_T(target_RT, cond_RT)
        # print(f"d_T: {d_T}")
        # continue
        save_path = f"outputs/vis_zero123_xl_finetune_scalar_real_{finetune_step}"

        os.makedirs(save_path, exist_ok=True)

        raw_im = cv2.imread(cond_img_path, cv2.IMREAD_GRAYSCALE)
        raw_im = 255 - raw_im
        cv2.imwrite(f"{save_path}/{frame_id:06d}_cam{source_cam}_raw_im.png", raw_im)

        gt_im = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        gt_im = 255 - gt_im
        cv2.imwrite(f"{save_path}/{frame_id:06d}_cam{target_cam}_gt_im.png", gt_im)

        raw_im = cv2.cvtColor(raw_im, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(raw_im)
        input_im = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = torchvision.transforms.functional.resize(input_im, [256, 256])

        out_imgs = main_run_simple(
            models,
            device,
            d_T,
            raw_im=input_im,
            save_path=save_path,
        )

        for i, img in enumerate(out_imgs):
            img.save(f"{save_path}/{frame_id:06d}_cam{source_cam}to{target_cam}_output_{i}.png")


if __name__ == "__main__":

    for ckp_id in range(15000, 15001, 1000):
        main_demo(finetune_step=ckp_id)
