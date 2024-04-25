"""
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
"""

import math
import sys

from contextlib import nullcontext
from functools import partial

import fire
import gradio as gr
import numpy as np
import torch

from camera_visualizer import CameraVisualizer
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from image_utils import preprocess_image
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms
from transformers import AutoFeatureExtractor  # , CLIPImageProcessor
from utils import load_model_from_config


_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
# _GPU_INDEX = 0
_GPU_INDEX = 3

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = "Zero-1-to-3: Zero-shot One Image to 3D Object"

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = """
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image.
It is based on Stable Diffusion. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
"""

_ARTICLE = "See uses.md"


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, x, y, z):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond["c_crossattn"] = [c]
            cond["c_concat"] = [
                model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)
            ]
            if scale != 1.0:
                uc = {}
                uc["c_concat"] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main_run(
    models,
    device,
    cam_vis,
    return_what,
    x=0.0,
    y=0.0,
    z=0.0,
    raw_im=None,
    preprocess=True,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    ddim_eta=1.0,
    precision="fp32",
    h=256,
    w=256,
    smoke=False,
    save_path=None,
):
    """
    :param raw_im (PIL Image).
    """

    # raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    # safety_checker_input = models["clip_fe"](raw_im, return_tensors="pt").to(device)
    # (image, has_nsfw_concept) = models["nsfw"](images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    # print("has_nsfw_concept:", has_nsfw_concept)
    # if np.any(has_nsfw_concept):
    #     print("NSFW content detected.")
    #     to_return = [None] * 10
    #     description = (
    #         '###  <span style="color:red"> Unfortunately, '
    #         "potential NSFW content was detected, "
    #         "which is not supported by our model. "
    #         "Please try again with a different image. </span>"
    #     )
    #     if "angles" in return_what:
    #         to_return[0] = 0.0
    #         to_return[1] = 0.0
    #         to_return[2] = 0.0
    #         to_return[3] = description
    #     else:
    #         to_return[0] = description
    #     return to_return

    # else:
    #     print("Safety check passed.")

    print("running main_run...")
    input_im = preprocess_image(models, raw_im, preprocess, smoke, save_path)

    # if np.random.rand() < 0.3:
    #     description = ('Unfortunately, a human, a face, or potential NSFW content was detected, '
    #                    'which is not supported by our model.')
    #     if vis_only:
    #         return (None, None, description)
    #     else:
    #         return (None, None, None, description)

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    if "rand" in return_what:
        x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))  # [-80, 80].
        y = int(np.round(np.random.uniform(-150.0, 150.0)))
        z = 0.0

    if cam_vis._gradio_plot is not None:
        cam_vis.polar_change(x)
        cam_vis.azimuth_change(y)
        cam_vis.radius_change(z)
        cam_vis.encode_image(show_in_im1)
        new_fig = cam_vis.update_figure()
    else:
        new_fig = None

    if "vis" in return_what:
        description = (
            "The viewpoints are visualized on the top right. "
            "Click Run Generation to update the results on the bottom right."
        )

        if "angles" in return_what:
            return (x, y, z, description, new_fig, show_in_im2)
        else:
            return (description, new_fig, show_in_im2)

    elif "gen" in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(models["turncam"])
        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(
            input_im, models["turncam"], sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, used_x, y, z
        )

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        description = None

        if "angles" in return_what:
            return (x, y, z, description, new_fig, show_in_im2, output_ims)
        else:
            return (description, new_fig, show_in_im2, output_ims)


def run_demo(
    device_idx=_GPU_INDEX,
    ckpt="/home/yuegao/Dynamics/zero123-weights/105000.ckpt",
    config="configs/sd-objaverse-finetune-c_concat-256.yaml",
):

    print("sys.argv:", sys.argv)
    if len(sys.argv) > 1:
        print("old device_idx:", device_idx)
        device_idx = int(sys.argv[1])
        print("new device_idx:", device_idx)

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print("Instantiating LatentDiffusion...")
    models["turncam"] = load_model_from_config(config, ckpt, device=device)
    print("Instantiating Carvekit HiInterface...")
    models["carvekit"] = create_carvekit_interface()
    print("Instantiating StableDiffusionSafetyChecker...")
    models["nsfw"] = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)
    print("Instantiating AutoFeatureExtractor...")
    models["clip_fe"] = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

    # Reduce NSFW false positives.
    # NOTE: At the time of writing, and for diffusers 0.12.1, the default parameters are:
    # models['nsfw'].concept_embeds_weights:
    # [0.1800, 0.1900, 0.2060, 0.2100, 0.1950, 0.1900, 0.1940, 0.1900, 0.1900, 0.2200, 0.1900,
    #  0.1900, 0.1950, 0.1984, 0.2100, 0.2140, 0.2000].
    # models['nsfw'].special_care_embeds_weights:
    # [0.1950, 0.2000, 0.2200].
    # We multiply all by some factor > 1 to make them less likely to be triggered.
    models["nsfw"].concept_embeds_weights *= 1.07
    models["nsfw"].special_care_embeds_weights *= 1.07

    with open("instructions.md", "r") as f:
        article = f.read()

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown("# " + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant="panel"):

                image_block = gr.Image(type="pil", image_mode="RGBA", label="Input image of single object")
                preprocess_chk = gr.Checkbox(
                    True, label="Preprocess image automatically (remove background and recenter object)"
                )
                # info='If enabled, the uploaded image will be preprocessed to remove the background and recenter the object by cropping and/or padding as necessary. '
                # 'If disabled, the image will be used as-is, *BUT* a fully transparent or white background is required.'),

                gr.Markdown("*Try camera position presets:*")
                with gr.Row():
                    left_btn = gr.Button("View from the Left", variant="primary")
                    above_btn = gr.Button("View from Above", variant="primary")
                    right_btn = gr.Button("View from the Right", variant="primary")
                with gr.Row():
                    random_btn = gr.Button("Random Rotation", variant="primary")
                    below_btn = gr.Button("View from Below", variant="primary")
                    behind_btn = gr.Button("View from Behind", variant="primary")

                gr.Markdown("*Control camera position manually:*")
                polar_slider = gr.Slider(-90, 90, value=0, step=5, label="Polar angle (vertical rotation in degrees)")
                # info='Positive values move the camera down, while negative values move the camera up.')
                azimuth_slider = gr.Slider(
                    -180, 180, value=0, step=5, label="Azimuth angle (horizontal rotation in degrees)"
                )
                # info='Positive values move the camera right, while negative values move the camera left.')
                radius_slider = gr.Slider(-0.5, 0.5, value=0.0, step=0.1, label="Zoom (relative distance from center)")
                # info='Positive values move the camera further away, while negative values move the camera closer.')

                samples_slider = gr.Slider(1, 8, value=4, step=1, label="Number of samples to generate")

                with gr.Accordion("Advanced options", open=False):
                    scale_slider = gr.Slider(0, 30, value=3, step=1, label="Diffusion guidance scale")
                    steps_slider = gr.Slider(5, 200, value=75, step=5, label="Number of diffusion inference steps")

                with gr.Row():
                    vis_btn = gr.Button("Visualize Angles", variant="secondary")
                    run_btn = gr.Button("Run Generation", variant="primary")

                desc_output = gr.Markdown("The results will appear on the right.", visible=_SHOW_DESC)

            with gr.Column(scale=1.1, variant="panel"):

                vis_output = gr.Plot(label="Relationship between input (green) and output (blue) camera poses")

                gen_output = gr.Gallery(label="Generated images from specified new viewpoint")
                gen_output.style(grid=2)

                preproc_output = gr.Image(
                    type="pil", image_mode="RGB", label="Preprocessed input image", visible=_SHOW_INTERMEDIATE
                )

        gr.Markdown(article)

        # NOTE: I am forced to update vis_output for these preset buttons,
        # because otherwise the gradio plot always resets the plotly 3D viewpoint for some reason,
        # which might confuse the user into thinking that the plot has been updated too.

        # OLD 1:
        # left_btn.click(fn=lambda: [0.0, -90.0], #, 0.0],
        #                inputs=[], outputs=[polar_slider, azimuth_slider]), #], radius_slider])
        # above_btn.click(fn=lambda: [90.0, 0.0], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # right_btn.click(fn=lambda: [0.0, 90.0], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # random_btn.click(fn=lambda: [int(np.round(np.random.uniform(-60.0, 60.0))),
        #                              int(np.round(np.random.uniform(-150.0, 150.0)))], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # below_btn.click(fn=lambda: [-90.0, 0.0], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # behind_btn.click(fn=lambda: [0.0, 180.0], #, 0.0],
        #                  inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])

        # OLD 2:
        # preset_text = ('You have selected a preset target camera view. '
        #                'Now click Run Generation to update the results!')

        # left_btn.click(fn=lambda: [0.0, -90.0, None, preset_text],
        #                inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # above_btn.click(fn=lambda: [90.0, 0.0, None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # right_btn.click(fn=lambda: [0.0, 90.0, None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # random_btn.click(fn=lambda: [int(np.round(np.random.uniform(-60.0, 60.0))),
        #                              int(np.round(np.random.uniform(-150.0, 150.0))),
        #                              None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # below_btn.click(fn=lambda: [-90.0, 0.0, None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # behind_btn.click(fn=lambda: [0.0, 180.0, None, preset_text],
        #                  inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])

        # OLD 3 (does not work at all):
        # def a():
        #     polar_slider.value = 77.7
        #     polar_slider.postprocess(77.7)
        #     print('testa')
        # left_btn.click(fn=a)

        cam_vis = CameraVisualizer(vis_output)

        vis_btn.click(
            fn=partial(main_run, models, device, cam_vis, "vis"),
            inputs=[polar_slider, azimuth_slider, radius_slider, image_block, preprocess_chk],
            outputs=[desc_output, vis_output, preproc_output],
        )

        run_btn.click(
            fn=partial(main_run, models, device, cam_vis, "gen"),
            inputs=[
                polar_slider,
                azimuth_slider,
                radius_slider,
                image_block,
                preprocess_chk,
                scale_slider,
                samples_slider,
                steps_slider,
            ],
            outputs=[desc_output, vis_output, preproc_output, gen_output],
        )

        # NEW:
        preset_inputs = [image_block, preprocess_chk, scale_slider, samples_slider, steps_slider]
        preset_outputs = [
            polar_slider,
            azimuth_slider,
            radius_slider,
            desc_output,
            vis_output,
            preproc_output,
            gen_output,
        ]
        left_btn.click(
            fn=partial(main_run, models, device, cam_vis, "angles_gen", 0.0, -90.0, 0.0),
            inputs=preset_inputs,
            outputs=preset_outputs,
        )
        above_btn.click(
            fn=partial(main_run, models, device, cam_vis, "angles_gen", -90.0, 0.0, 0.0),
            inputs=preset_inputs,
            outputs=preset_outputs,
        )
        right_btn.click(
            fn=partial(main_run, models, device, cam_vis, "angles_gen", 0.0, 90.0, 0.0),
            inputs=preset_inputs,
            outputs=preset_outputs,
        )
        random_btn.click(
            fn=partial(main_run, models, device, cam_vis, "rand_angles_gen", -1.0, -1.0, -1.0),
            inputs=preset_inputs,
            outputs=preset_outputs,
        )
        below_btn.click(
            fn=partial(main_run, models, device, cam_vis, "angles_gen", 90.0, 0.0, 0.0),
            inputs=preset_inputs,
            outputs=preset_outputs,
        )
        behind_btn.click(
            fn=partial(main_run, models, device, cam_vis, "angles_gen", 0.0, 180.0, 0.0),
            inputs=preset_inputs,
            outputs=preset_outputs,
        )

    demo.launch(enable_queue=True, share=True, server_name="0.0.0.0")


if __name__ == "__main__":

    fire.Fire(run_demo)
