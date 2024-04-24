import os
import shutil

logs_dir = "logs"
exp_folders = sorted(os.listdir(logs_dir))

for exp_folder in exp_folders:
    ckp_path = os.path.join(logs_dir, exp_folder, "checkpoints")
    files = os.listdir(ckp_path)
    if len(files) > 0:
        continue
    print(f"Deleting empty experiment: {exp_folder}")
    shutil.rmtree(os.path.join(logs_dir, exp_folder))
