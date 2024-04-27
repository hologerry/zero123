import os

checkpoints_path = "logs/2024-04-24T14-04-16_sd-scalar-flow-finetune-c_concat-256/checkpoints"
checkpoints_names = os.listdir(checkpoints_path)

for name in checkpoints_names:
    if "step=" in name:
        step_name = name.split("-")[1]
        new_name = f"{step_name}"
        print(f"Renamed {name} to {new_name}")
        os.rename(f"{checkpoints_path}/{name}", f"{checkpoints_path}/{new_name}")

