import json
import math
import os

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from camera_utils import get_T
from demo_helpers import main_run_simple
from loss_utils import l1_loss as l1
from loss_utils import psnr, ssim
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from skimage.metrics import structural_similarity as sk_ssim
from tqdm import tqdm
from utils import load_model_from_config


def main_eval(device_idx=0, finetune_step=6000):
    # ckpt = "/home/yuegao/Dynamics/stable-zero123/stable_zero123.ckpt"
    # ckpt = "/home/yuegao/Dynamics/zero123-weights/zero123-xl.ckpt"
    # ckpt = "/home/yuegao/Dynamics/zero123-weights/165000.ckpt"
    # ckpt = "logs/2024-04-24T02-19-47_sd-scalar-flow-finetune-c_concat-256/checkpoints/last.ckpt"
    ckpt = f"logs/2024-04-24T14-04-16_sd-scalar-flow-finetune-c_concat-256/checkpoints/step={finetune_step-1:09d}.ckpt"
    # config = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    config = "configs/sd-scalar-flow-finetune-c_concat-256.yaml"

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print("Instantiating LatentDiffusion...")
    models["turncam"] = load_model_from_config(config, ckpt, device=device)

    root_dir = "/data/Dynamics/ScalarFlow/zero123_dataset/"
    with open(os.path.join(root_dir, "valid_paths.json")) as f:
        paths = json.load(f)

    total_objects = len(paths)
    val_paths = paths[math.floor(total_objects / 100.0 * 99.0) :]  # used last 1% as validation

    # sim_frame_name = "sim_000103_frame_000100"
    source_cam = "02"
    # target_cam = "00"

    psnrs = []
    lpipss = []
    lpipss_vggs = []

    l1s = []
    ssims = []
    ssims_v2 = []

    lpips_criteria = lpips.LPIPS(net="alex").to(device)
    lpips_vgg_criteria = lpips.LPIPS(net="vgg").to(device)

    for sim_frame_name in tqdm(val_paths):
        for target_cam in ["00", "01", "03", "04"]:
            cond_img_path = f"/data/Dynamics/ScalarFlow/zero123_dataset/{sim_frame_name}/{source_cam}.png"
            gt_img_path = f"/data/Dynamics/ScalarFlow/zero123_dataset/{sim_frame_name}/{target_cam}.png"
            assert os.path.exists(cond_img_path), f"cond_img_path {cond_img_path} does not exist"
            assert os.path.exists(gt_img_path), f"gt_img_path {gt_img_path} does not exist"

            cond_cam_path = f"/data/Dynamics/ScalarFlow/zero123_dataset/camera/{source_cam}.npy"
            target_cam_path = f"/data/Dynamics/ScalarFlow/zero123_dataset/camera/{target_cam}.npy"
            cond_RT = np.load(cond_cam_path)
            target_RT = np.load(target_cam_path)

            d_T = get_T(target_RT, cond_RT)

            save_path = "outputs/vis_zero123_xl_finetune"

            os.makedirs(save_path, exist_ok=True)

            raw_im = cv2.imread(cond_img_path, cv2.IMREAD_GRAYSCALE)
            raw_im = 255 - raw_im
            cv2.imwrite(f"{save_path}/{sim_frame_name}_cam{source_cam}_raw_im.png", raw_im)

            gt_im = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
            gt_im = 255 - gt_im
            cv2.imwrite(f"{save_path}/{sim_frame_name}_cam{target_cam}_gt_im.png", gt_im)

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
                save_path=save_path,
                n_samples=1,
            )

            # for i, img in enumerate(out_imgs):
            #     img.save(f"{save_path}/{sim_frame_name}_cam{source_cam}to{target_cam}_output_{i}.png")

            out_img = out_imgs[0]
            output_im = torchvision.transforms.ToTensor()(out_img).unsqueeze(0).to(device)
            output_im = output_im * 2 - 1

            ssims.append(ssim(output_im, target_im))
            l1s.append(l1(output_im, target_im))

            psnrs.append(psnr(output_im, target_im))
            lpipss.append(lpips_criteria(output_im, target_im))
            lpipss_vggs.append(lpips_vgg_criteria(output_im, target_im))

            render_numpy = output_im[0].permute(1, 2, 0).detach().cpu().numpy()
            gt_numpy = target_im[0].permute(1, 2, 0).detach().cpu().numpy()
            ssim_v2 = sk_ssim(
                render_numpy, gt_numpy, channel_axis=-1, multichannel=True, data_range=gt_numpy.max() - gt_numpy.min()
            )
            ssims_v2.append(ssim_v2)

    output_dict = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        "L1": torch.tensor(l1s).mean().item(),
        "SSIM_v2": torch.tensor(ssims_v2).mean().item(),
        "LPIPS_VGG": torch.tensor(lpipss_vggs).mean().item(),
    }
    output_json = f"outputs/eval_zero123_xl_finetune_2024-04-24T14-04-16_{finetune_step}.json"
    with open(output_json, "w") as f:
        json.dump(output_dict, f)


if __name__ == "__main__":

    for ckp_id in range(15500, 15501, 1):
        main_eval(device_idx=0, finetune_step=ckp_id)
