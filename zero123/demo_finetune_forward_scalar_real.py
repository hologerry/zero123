import argparse
import math
import os

import cv2
import lpips
import numpy as np
import torch
import torchvision

from camera_utils import get_T
from demo_helpers import main_run_simple
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from utils import load_model_from_config


def main_demo(source_cam=0, device_idx=0, finetune_step=1000):
    print(f"source_cam: {source_cam}, device_idx: {device_idx}, finetune_step: {finetune_step}")
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
    out_root = f"/data/Dynamics/ScalarReal/zero123_finetune_{finetune_step}"

    lpips_criteria = lpips.LPIPS(net="alex").to(device)

    cam_ids = [i for i in range(5)]
    frame_ids = [i for i in range(120)]

    for target_cam in cam_ids:
        if source_cam == target_cam:
            continue
        for frame_id in frame_ids:
            out_scalar_real_zero123_dataset_path = f"{out_root}_cam{source_cam}to{target_cam}"
            os.makedirs(out_scalar_real_zero123_dataset_path, exist_ok=True)

            cond_img_path = f"{scalar_real_zero123_dataset_path}/frame_{frame_id:06d}/{source_cam:02d}.png"
            gt_img_path = f"{scalar_real_zero123_dataset_path}/frame_{frame_id:06d}/{target_cam:02d}.png"
            assert os.path.exists(cond_img_path), f"cond_img_path {cond_img_path} does not exist"
            assert os.path.exists(gt_img_path), f"gt_img_path {gt_img_path} does not exist"

            cond_cam_path = f"{scalar_real_zero123_dataset_path}/camera/{source_cam:02d}.npy"
            target_cam_path = f"{scalar_real_zero123_dataset_path}/camera/{target_cam:02d}.npy"
            cond_RT = np.load(cond_cam_path)
            target_RT = np.load(target_cam_path)

            d_T = get_T(target_RT, cond_RT)
            # print(f"{source_cam} to {target_cam} d_T: {d_T}")
            # continue

            raw_im = cv2.imread(cond_img_path, cv2.IMREAD_GRAYSCALE)
            raw_im = 255 - raw_im
            # cv2.imwrite(f"{save_path}/{sim_frame_name}_cam{source_cam}_raw_im.png", raw_im)

            gt_im = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
            gt_im = 255 - gt_im
            # cv2.imwrite(f"{save_path}/{sim_frame_name}_cam{target_cam}_gt_im.png", gt_im)

            raw_im = cv2.cvtColor(raw_im, cv2.COLOR_GRAY2RGB)
            raw_image = Image.fromarray(raw_im)
            input_im = torchvision.transforms.ToTensor()(raw_image).unsqueeze(0).to(device)
            input_im = input_im * 2 - 1
            input_im = torchvision.transforms.functional.resize(input_im, [256, 256])

            gt_im = cv2.cvtColor(gt_im, cv2.COLOR_GRAY2RGB)
            gt_image = Image.fromarray(gt_im)
            target_im = torchvision.transforms.ToTensor()(gt_image).unsqueeze(0).to(device)
            target_im = target_im * 2 - 1
            target_im = torchvision.transforms.functional.resize(target_im, [256, 256])

            out_imgs = main_run_simple(
                models,
                device,
                d_T,
                raw_im=input_im,
            )

            min_lpips = math.inf
            best_img = None
            for i, img in enumerate(out_imgs):
                # img.save(f"{save_path}/{sim_frame_name}_cam{source_cam}to{target_cam}_output_{i}.png")

                output_im = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
                output_im = output_im * 2 - 1
                cur_lpips = lpips_criteria(output_im, target_im).item()
                if cur_lpips < min_lpips:
                    min_lpips = cur_lpips
                    best_img = img

            save_path = f"{out_scalar_real_zero123_dataset_path}/frame_{frame_id:06d}.png"
            best_img.save(save_path)


if __name__ == "__main__":

    main_demo(source_cam=4, finetune_step=15000)
