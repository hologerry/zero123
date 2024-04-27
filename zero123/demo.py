import os

import cv2

from camera_visualizer import CameraVisualizer
from demo_helpers import main_run_smoke
from gradio_new import main_run
from omegaconf import OmegaConf
from rich import print
from tqdm import tqdm
from utils import load_model_from_config


def main_demo():
    device_idx = 0
    # ckpt = "/home/yuegao/Dynamics/stable-zero123/stable_zero123.ckpt"
    ckpt = "/home/yuegao/Dynamics/zero123-weights/zero123-xl.ckpt"
    # ckpt = "/home/yuegao/Dynamics/zero123-weights/165000.ckpt"
    config = "configs/sd-objaverse-finetune-c_concat-256.yaml"

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print("Instantiating LatentDiffusion...")
    models["turncam"] = load_model_from_config(config, ckpt, device=device)

    cam_vis = CameraVisualizer(None)

    # img_path = "/home/yuegao/Dynamics/HyFluid/data/ScalarReal/colmap_100/input/train02.png"
    img_path = "/data/Dynamics/HyFluid_data/real_smoke_231026/view3_stable_square/0120.jpg"
    ## stable video diffusion preprocessd with putting object to the center
    img_path = "/home/yuegao/Dynamics/generative-models/outputs/simple_video_sample/sv3d_p/000001_input.jpg"
    raw_im = cv2.imread(img_path)
    raw_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2RGB)
    raw_im = cv2.resize(raw_im, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    save_path = "outputs/vis_hyfluid_teaser_view1_zero123_xl"
    os.makedirs(save_path, exist_ok=True)

    print(f"raw_im shape: {raw_im.shape}")
    cv2.imwrite(f"{save_path}/input_im.png", raw_im)

    y_angle_values = [float(i) for i in range(-45, 50, 5)]

    for idx, y in tqdm(enumerate(y_angle_values)):
        outputs = main_run_smoke(
            models,
            device,
            cam_vis,
            "trans_gen",
            0.0,
            y,
            0.0,
            raw_im=raw_im,
            preprocess=False,
            smoke=True,
            save_path=save_path,
        )
        # for o in outputs:
        #     print(type(o))

        out_imgs = outputs[-1]
        for i, img in enumerate(out_imgs):
            img.save(f"{save_path}/idx_{idx}_y{y}_output_{i}.png")


if __name__ == "__main__":

    main_demo()
