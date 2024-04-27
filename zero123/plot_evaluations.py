import json
import os

import matplotlib.pyplot as plt
import pandas as pd


# Path to the directory containing JSON files
json_directory = "outputs"
all_json_files = sorted(os.listdir(json_directory))
all_json_files = [f for f in all_json_files if f.endswith(".json")]
prefix = "eval_zero123_xl_finetune_2024-04-24T14-04-16_"
metrics = ['SSIM', 'PSNR', 'LPIPS', 'L1', 'SSIM_v2', 'LPIPS_VGG']

# Load all JSON files into pandas DataFrames
data_frames = []
all_scores_dict = {}
for json_name in all_json_files:
    file_path = os.path.join(json_directory, json_name)
    trainsteps = json_name[:-5][len(prefix) :]
    with open(file_path, "r") as file:
        data = json.load(file)
    all_scores_dict[trainsteps] = data

all_scores_dict = sorted(all_scores_dict.items(), key=lambda x: int(x[0]))

