import json
import os


sub_data_path = "/data/Dynamics/Zero123_Objaverse/views_release_sub"

out_json_path = f"{sub_data_path}/valid_paths.json"

path_names = sorted(os.listdir(sub_data_path))[:100]


with open(out_json_path, "w") as f:
    json.dump(path_names, f)
